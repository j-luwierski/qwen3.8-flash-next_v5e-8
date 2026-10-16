"""Deterministic independent workers and verified offline artifact exchange.

Workers share no model state and need no incoming network connection. Their
outputs can be transferred as Kaggle Dataset artifacts, then merged locally.
"""

import fcntl
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .devices import CaptureReplicas
from .io import atomic_write, digest, read_json, sha256, write_json
from .runtime import check, SessionPaused, resumable_copy


def normalize_record(record):
    ids = record.get("input_ids")
    mask = record.get("loss_mask")
    if not isinstance(ids, list) or len(ids) < 2 or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError("Every sample needs input_ids: a list of >=2 nonnegative integers")
    if not isinstance(mask, list) or len(mask) != len(ids) or any(type(t) not in (int, float) or t not in (0, 1) for t in mask):
        raise ValueError("Every sample needs a binary loss_mask with the same length as input_ids")
    if not any(a and b for a, b in zip(mask, mask[1:])):
        raise ValueError("DFlash requires at least two consecutive supervised tokens")
    value = {"input_ids": ids, "loss_mask": [int(x) for x in mask]}
    value["id"] = str(record.get("id", digest(value)))
    return value


def make_plan(data, out, workers, *, tokenizer_identity, kind="tokens"):
    if workers < 1:
        raise ValueError("workers must be positive")
    out = Path(out)
    if out.exists():
        raise ValueError(f"Plan directory already exists: {out}")
    from .data import normalize_prompt
    if kind not in ("tokens", "prompts"):
        raise ValueError("Plan kind must be tokens or prompts")
    normalize = normalize_record if kind == "tokens" else normalize_prompt
    records = [normalize(json.loads(line)) for line in Path(data).read_text().splitlines() if line.strip()]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Dataset must be nonempty and sample ids must be unique")
    if workers > len(records):
        raise ValueError("More workers than records would leave empty shards")
    items = [{"key": digest(r["id"]), "record_hash": digest(r), "worker": i % workers}
             for i, r in enumerate(records)]
    payload = {"schema": 1, "workers": workers, "tokenizer_identity": tokenizer_identity, "items": items}
    if kind == "prompts":
        payload["kind"] = kind
    plan = {**payload, "plan_id": digest(payload)}
    out.mkdir(parents=True)
    for worker in range(workers):
        part = [r for i, r in enumerate(records) if i % workers == worker]
        atomic_write(out / f"part-{worker:04d}.jsonl",
                     lambda f, part=part: f.write("".join(json.dumps(r) + "\n" for r in part).encode()))
    write_json(out / "plan.json", plan)
    return plan


def read_plan(directory):
    plan = read_json(Path(directory) / "plan.json")
    if digest({k: v for k, v in plan.items() if k != "plan_id"}) != plan["plan_id"]:
        raise ValueError("Plan checksum mismatch")
    return plan


def validate_features(record, contract, length, mode="hidden_states"):
    required = {"input_ids", "loss_mask"}
    if mode == "hidden_states":
        required |= {"hidden_states", "target_last_hidden_states"}
    elif mode != "tokens":
        raise ValueError("Unknown feature mode")
    if set(record) != required or any(not isinstance(v, torch.Tensor) for v in record.values()):
        raise ValueError("Unexpected offline feature schema")
    h = contract["hidden_size"]
    shapes = {"input_ids": (length,), "loss_mask": (length,),
              "hidden_states": (1, length, h * len(contract["layer_ids"])),
              "target_last_hidden_states": (1, length, h)}
    for key, shape in shapes.items():
        if key not in required:
            continue
        if tuple(record[key].shape) != shape or not torch.isfinite(record[key]).all():
            raise ValueError(f"Invalid feature {key}: expected finite tensor {shape}")
    if record["input_ids"].dtype != torch.long:
        raise ValueError("input_ids must be int64")


def capture_worker(plan_dir, worker, model_path, out, layers, *, revision=None, dtype="float32", device="cpu", backend="torch",
                   devices="all", tokenizer_path=None, template_kwargs=None, generation=None,
                   max_length=None, feature_mode="hidden_states", seed=42, engine_options=None, control=None):
    check(control, 'capture')
    plan = read_plan(plan_dir)
    if feature_mode not in ("hidden_states", "tokens"):
        raise ValueError("Unknown feature mode")
    if not 0 <= worker < plan["workers"]:
        raise ValueError("worker index out of range")
    entries = [item for item in plan["items"] if item["worker"] == worker]
    from .data import ChatTokenizer, normalize_prompt
    is_prompts = plan.get("kind") == "prompts"
    normalize = normalize_prompt if is_prompts else normalize_record
    records = [normalize(json.loads(line)) for line in
               (Path(plan_dir) / f"part-{worker:04d}.jsonl").read_text().splitlines()]
    if [digest(r) for r in records] != [e["record_hash"] for e in entries]:
        raise ValueError("Shard data differs from the plan")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Lock is held for the lifetime of this worker, including model loading.
    with open(out / ".worker.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with CaptureReplicas(model_path, layers, revision=revision, dtype=dtype,
                             device=device, backend=backend, devices=devices,
                             **((engine_options or {}) if backend == "jax" else
                                {k: v for k, v in (engine_options or {}).items() if k == 'config_identity_path'})) as replicas:
            contract = {**replicas.contract, "tokenizer_identity": plan["tokenizer_identity"]}
            tokenizer = None
            length = max_length or replicas.engines[0].config.max_position_embeddings
            if length < 2 or length > replicas.engines[0].config.max_position_embeddings:
                raise ValueError("max_length must be between 2 and the target context limit")
            if is_prompts:
                tokenizer = ChatTokenizer(tokenizer_path or model_path, revision=revision, template_kwargs=template_kwargs)
                if tokenizer.identity != plan["tokenizer_identity"]:
                    raise ValueError("Tokenizer/chat-template identity differs from the prompt plan")
                contract["generation"] = {"options": generation or {}, "max_length": length, "seed": seed}
            if feature_mode != "hidden_states":
                contract["feature_mode"] = feature_mode
            identity = {"plan_id": plan["plan_id"], "worker": worker, "contract": contract}
            identity_path = out / "identity.json"
            if identity_path.exists() and read_json(identity_path) != identity:
                raise ValueError("Cannot resume worker with a different target, plan, dtype or capture policy")
            write_json(identity_path, identity)
            # Check completed files centrally, then distribute only missing work.
            # A resumed worker can use a different number of cores safely.
            results, pending = {}, []
            for index, (entry, sample) in enumerate(zip(entries, records)):
                key = entry["key"]
                target = out / "features" / f"{key}.ckpt"
                receipt = out / "receipts" / f"{key}.json"
                if receipt.exists():
                    saved = read_json(receipt)
                    if saved["record_hash"] != entry["record_hash"] or (
                            saved.get("status") != "skipped" and sha256(target) != saved["sha256"]):
                        raise ValueError(f"Corrupted completed feature: {key}")
                    if is_prompts and sha256(out / "conversations" / f"{key}.json") != saved.get("conversation_sha256"):
                        raise ValueError(f"Corrupted completed conversation: {key}")
                    results[index] = saved
                else:
                    pending.append((index, entry, sample))

            def capture_partition(engine, partition):
                completed = {}
                for index, entry, sample in partition:
                    check(control, 'capture')
                    target = out / "features" / f"{entry['key']}.ckpt"
                    receipt = out / "receipts" / f"{entry['key']}.json"
                    conversation_hash = {}
                    if tokenizer is not None:
                        transcript = out / "conversations" / f"{entry['key']}.json"
                        if transcript.exists():
                            # An interrupted sample can have a transcript but no
                            # receipt yet. Its own envelope binds it to this job.
                            envelope = read_json(transcript)
                            generated = envelope["generated"]
                            if envelope["input"] != digest([identity, entry]) or envelope["checksum"] != digest(generated):
                                raise ValueError("Corrupted or mismatched pending conversation")
                        else:
                            partial = out / 'generation' / f"{entry['key']}.json"
                            progress = None
                            if partial.exists():
                                envelope = read_json(partial)
                                progress = envelope['progress']
                                if envelope['input'] != digest([identity, entry]) or envelope['checksum'] != digest(progress):
                                    raise ValueError('Corrupted or mismatched generation checkpoint')
                            def save_generation(value):
                                write_json(partial, dict(input=digest([identity, entry]), progress=value, checksum=digest(value)))
                            generated = tokenizer.example(engine, sample, max_length=length,
                                                          generation=generation or {},
                                                          seed=int(digest([seed, entry["key"]])[:8], 16),
                                                          progress=progress, on_progress=save_generation, control=control)
                            write_json(transcript, {"generated": generated, "input": digest([identity, entry]),
                                                    "checksum": digest(generated)})
                        conversation_hash = {"conversation_sha256": sha256(transcript)}
                        if generated["status"] == "skipped":
                            saved = {**entry, **generated, **conversation_hash}
                            write_json(receipt, saved)
                            completed[index] = saved
                            continue
                        sample = normalize_record(generated)
                    check(control, 'capture')
                    if feature_mode == "tokens":
                        features = {"input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
                                    "loss_mask": torch.tensor(sample["loss_mask"], dtype=torch.float32)}
                    else:
                        features = engine.capture(sample["input_ids"], sample["loss_mask"])
                        if dtype == "bfloat16":
                            for key in ("hidden_states", "target_last_hidden_states"):
                                features[key] = features[key].to(torch.bfloat16)
                    validate_features(features, contract, len(sample["input_ids"]), feature_mode)
                    atomic_write(target, lambda f: torch.save(features, f))
                    saved = {**entry, **conversation_hash, "sha256": sha256(target), "length": len(sample["input_ids"])}
                    write_json(receipt, saved)
                    completed[index] = saved
                return completed

            partitions = [pending[index::replicas.count] for index in range(replicas.count)]
            print(f"Capture worker {worker}: {replicas.count} local {device} devices, "
                  f"{len(pending)} pending / {len(results)} resumed samples", flush=True)
            with ThreadPoolExecutor(max_workers=replicas.count, thread_name_prefix="qwen-device") as pool:
                futures = [pool.submit(capture_partition, engine, partition)
                           for engine, partition in zip(replicas.engines, partitions)]
                # Exiting the executor waits for every device before readers close,
                # also when a device fails; completed receipts remain resumable.
                paused = None
                for future in futures:
                    try:
                        results.update(future.result())
                    except SessionPaused as exc:
                        paused = exc
                if paused:
                    write_json(out / 'progress.json', dict(status='paused', total=len(entries),
                        completed=len(list((out / 'receipts').glob('*.json'))), stage=paused.stage))
                    raise paused
            assets = assets_config = None
            if worker == 0:
                check(control, 'teacher_assets')
                replicas.engines[0].save_assets(out / "assets")
                assets = sha256(out / "assets" / "teacher.safetensors")
                assets_config = sha256(out / "assets" / "config.json")
            manifest = {**identity, "items": [results[index] for index in range(len(entries))],
                        "assets_sha256": assets, "assets_config_sha256": assets_config,
                        "feature_mode": feature_mode,
                        "execution": {"devices": replicas.device_names,
                                      "weight_cache": [e.weight_cache.stats() for e in replicas.engines] if backend == "jax" else [],
                                      "samples_per_device": [len(partition) for partition in partitions],
                                      "resumed_samples": len(entries) - len(pending)}}
            write_json(out / "manifest.json", manifest)
            return manifest


def merge_workers(plan_dir, worker_dirs, out, *, control=None):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_name(out.name + ".merge.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _merge_workers(plan_dir, worker_dirs, out, control=control)


def _merge_workers(plan_dir, worker_dirs, out, *, control=None):
    plan = read_plan(plan_dir)
    out = Path(out)
    if out.exists():
        raise ValueError(f"Merge output already exists: {out}")
    expected = {entry["key"]: entry for entry in plan["items"]}
    seen, workers, contract, assets_dir, assets_hash, config_hash = {}, set(), None, None, None, None
    skipped, feature_mode, conversations = {}, None, {}
    for directory in map(Path, worker_dirs):
        manifest = read_json(directory / "manifest.json")
        worker = manifest["worker"]
        if manifest["plan_id"] != plan["plan_id"] or worker in workers:
            raise ValueError("Wrong plan or duplicate worker output")
        if not 0 <= worker < plan["workers"]:
            raise ValueError("Invalid worker number")
        workers.add(worker)
        contract = contract or manifest["contract"]
        if contract != manifest["contract"]:
            raise ValueError("Workers used incompatible target/capture contracts")
        mode = manifest.get("feature_mode", "hidden_states")
        feature_mode = feature_mode or mode
        if feature_mode != mode:
            raise ValueError("Workers mixed token-only and hidden-state features")
        for item in manifest["items"]:
            key = item["key"]
            if item["worker"] != worker or key in seen or key in skipped or key not in expected or any(item[k] != expected[key][k] for k in expected[key]):
                raise ValueError("Duplicated, misplaced or unexpected sample")
            if "conversation_sha256" in item:
                transcript = directory / "conversations" / f"{key}.json"
                conversations[key] = (transcript, item["conversation_sha256"])
            if item.get("status") == "skipped":
                skipped[key] = item
                continue
            path = directory / "features" / f"{key}.ckpt"
            seen[key] = (path, item)
        if worker == 0:
            assets_dir = directory / "assets"
            assets_hash = manifest["assets_sha256"]
            config_hash = manifest["assets_config_sha256"]
            asset_contract = read_json(assets_dir / "contract.json")
            if any(contract[k] != v for k, v in asset_contract.items()):
                raise ValueError("Teacher assets do not match captured features")
    if set(seen) | set(skipped) != set(expected) or workers != set(range(plan["workers"])) or assets_dir is None:
        raise ValueError("Missing workers or samples: refusing incomplete training corpus")
    if not seen:
        raise ValueError("All prompts were skipped; no trainable samples")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.with_name(out.name + ".merging")
    manifest = {"plan_id": plan["plan_id"], "contract": contract,
                "assets_sha256": assets_hash, "assets_config_sha256": config_hash,
                "feature_mode": feature_mode, "skipped": list(skipped.values()),
                "items": [seen[e["key"]][1] for e in plan["items"] if e["key"] in seen]}
    intended = digest(manifest)
    request = staging / 'merge.request.json'
    if staging.exists():
        if not request.exists():
            shutil.rmtree(staging)  # Older, unpublished staging format.
        elif read_json(request)['identity'] != intended:
            raise ValueError('Pending merge uses a different plan or worker artifacts')
    staging.mkdir(exist_ok=True)
    write_json(request, dict(identity=intended))
    try:
        for key, (path, checksum) in sorted(conversations.items()):
            check(control, 'merge')
            resumable_copy(path, staging / 'conversations' / f'{key}.json', checksum, control)
        completed = 0
        for key in sorted(seen):
            check(control, 'merge')
            path, item = seen[key]
            target = staging / 'features' / f'{key}.ckpt'
            if not target.exists():
                features = torch.load(path, map_location='cpu', weights_only=True)
                validate_features(features, contract, item['length'], feature_mode)
                del features
            resumable_copy(path, target, item['sha256'], control)
            completed += 1
            write_json(staging / 'progress.json', dict(status='running', completed=completed, total=len(seen)))
        for name, checksum in (('teacher.safetensors', assets_hash), ('config.json', config_hash),
                               ('contract.json', sha256(assets_dir / 'contract.json'))):
            check(control, 'merge_assets')
            resumable_copy(assets_dir / name, staging / 'assets' / name, checksum, control)
        if (assets_dir / 'origin.json').exists():
            resumable_copy(assets_dir / 'origin.json', staging / 'assets/origin.json',
                           sha256(assets_dir / 'origin.json'), control)
        write_json(staging / 'manifest.json', manifest)
        write_json(staging / 'progress.json', dict(status='complete', completed=len(seen), total=len(seen)))
        os.rename(staging, out)
    except SessionPaused:
        raise  # Keep verified files and unfinished byte prefixes for resume.
    except BaseException:
        shutil.rmtree(staging)
        raise
    return manifest
