"""A resumable job: public/local prompts -> teacher responses -> corpus -> draft."""

import fcntl
import json
import random
import shutil
import tempfile
import zipfile
from contextlib import ExitStack
from pathlib import Path

from .data import ChatTokenizer, fetch_prompts
from .io import atomic_write, digest, read_json, sha256, write_json
from .shards import make_plan, read_plan, capture_worker, merge_workers
from .training import train
from .runtime import RunControl, SessionPaused, check
from .session_io import unpack_cached, snapshot_job, restore_job


def pack_worker(directory, out):
    directory, out = Path(directory).resolve(), Path(out).resolve()
    if out.is_relative_to(directory):
        raise ValueError("Archive must be outside the worker directory")
    def write(file):
        with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in sorted(directory.rglob("*")):
                if path.is_file() and not path.name.endswith((".tmp", ".lock")):
                    archive.write(path, str(path.relative_to(directory)))
    atomic_write(out, write)
    write_json(out.with_suffix(".sha256.json"), {"sha256": sha256(out)})
    return out


def artifact_directory(path, stack, *, cache_root=None, control=None):
    path = Path(path)
    if path.is_dir():
        return path
    if cache_root is not None:
        return unpack_cached(path, Path(cache_root) / sha256(path), control)
    temporary = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="qwen-worker-")))
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not (temporary / info.filename).resolve().is_relative_to(temporary):
                raise ValueError("Unsafe worker archive path")
            if info.external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError("Worker archives cannot contain symbolic links")
        archive.extractall(temporary)
    return temporary


def prepare_job(config, control=None):
    check(control, 'prepare')
    root = Path(config["out"])
    root.mkdir(parents=True, exist_ok=True)
    requested = {"data": config["data"], "workers": config["workers"],
                 "template_kwargs": config.get("chat_template_kwargs", {})}
    if config["data"].get("data"):
        requested["local_data_sha256"] = sha256(config["data"]["data"])
    tokenizer = ChatTokenizer(config.get("tokenizer", config["model"]), config.get("revision"),
                               config.get("chat_template_kwargs"))
    requested["tokenizer"] = tokenizer.identity
    def portable(value):
        value = json.loads(json.dumps(value))
        if value.get('local_data_sha256'):
            value['data']['data'] = '<local-file-identified-by-sha256>'
        return value
    requested = portable(requested)
    metadata = root / "preparation.json"
    with open(root / ".prepare.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if metadata.exists():
            if portable(read_json(metadata)["requested"]) != requested:
                raise ValueError("Job inputs changed; use a new output directory")
            return root
        request_path = root / "preparation.request.json"
        if request_path.exists() and portable(read_json(request_path)) != requested:
            raise ValueError("Job inputs changed during an interrupted preparation")
        write_json(request_path, requested)
        data = dict(config["data"])
        fraction = data.pop("validation_fraction", .05)
        if not 0 <= fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        prompts_path = root / "prompts.jsonl"
        source_metadata = prompts_path.with_suffix(".source.json")
        if prompts_path.exists() and source_metadata.exists():
            if sha256(prompts_path) != read_json(source_metadata)["sha256"]:
                raise ValueError("Cached prompts checksum differs")
            rows = [json.loads(line) for line in prompts_path.read_text().splitlines()]
        else:
            prompts_path.unlink(missing_ok=True)
            rows = fetch_prompts(prompts_path, control=control, **data)
        order = list(range(len(rows)))
        random.Random(data.get("seed", 42)).shuffle(order)
        count = max(1, int(len(rows) * fraction)) if fraction and len(rows) > 1 else 0
        validation = set(order[:count])
        for name, subset in (("plan", [r for i, r in enumerate(rows) if i not in validation]),
                             ("validation-plan", [r for i, r in enumerate(rows) if i in validation])):
            if not subset:
                continue
            check(control, 'prepare_plan')
            source = root / f"{name}.jsonl"
            atomic_write(source, lambda f: f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset).encode()))
            workers = min(config["workers"], len(subset))
            staging = root / (name + ".preparing")
            if staging.exists():
                shutil.rmtree(staging)
            intended = make_plan(source, staging, workers, tokenizer_identity=tokenizer.identity, kind="prompts")
            if (root / name).exists():
                if read_plan(root / name) != intended:
                    raise ValueError("Existing plan differs from intended inputs")
                shutil.rmtree(staging)
            else:
                staging.rename(root / name)
        write_json(metadata, {"requested": requested, "plan_id": read_plan(root / "plan")["plan_id"],
                              "source": read_json(prompts_path.with_suffix(".source.json"))})
        return root


def _run_job(config, *, stage="all", worker=0, artifacts=None, control=None):
    root = prepare_job(config, control)
    if stage in ("all", "worker"):
        worker_root = root / f"worker-{worker}"
        for name, output in (("plan", "train"), ("validation-plan", "validation")):
            if not (root / name).exists():
                continue
            plan = read_plan(root / name)
            if worker >= plan["workers"]:
                if name == "plan":
                    raise ValueError("worker index exceeds the prepared plan's worker count")
                continue
            capture_worker(root / name, worker, config["model"], worker_root / output, config["layers"],
                backend=config.get("backend", "jax"), device=config.get("device", "cpu"),
                devices=config.get("devices", "all"), dtype=config.get("dtype", "bfloat16"),
                revision=config.get("revision"), tokenizer_path=config.get("tokenizer"),
                template_kwargs=config.get("chat_template_kwargs"), generation=config.get("generation", {}),
                max_length=config.get("max_length", 4096), feature_mode=config.get("feature_mode", "tokens"),
                seed=config.get("seed", 42), engine_options=config.get("engine_options"), control=control)
        if config.get("pack_worker", False):
            check(control, 'pack_worker')
            pack_worker(worker_root, root / f"worker-{worker}.zip")
        if stage == "worker":
            return worker_root
    if stage in ("all", "train", "merge"):
        paths = artifacts or [root / f"worker-{i}" for i in range(read_plan(root / "plan")["workers"])]
        with ExitStack() as stack:
            needs_merge = any((root / name).exists() and not (root / output).exists()
                              for name, output in (("plan", "corpus"), ("validation-plan", "validation-corpus")))
            directories = [artifact_directory(path, stack, cache_root=root / 'artifact-cache', control=control)
                           for path in paths] if needs_merge else []
            for name, output, subset in (("plan", "corpus", "train"), ("validation-plan", "validation-corpus", "validation")):
                if not (root / name).exists():
                    continue
                if (root / output).exists():
                    if read_json(root / output / "manifest.json")["plan_id"] != read_plan(root / name)["plan_id"]:
                        raise ValueError("Merged corpus belongs to another plan")
                    continue
                available = [path / subset for path in directories if (path / subset / "manifest.json").exists()]
                merge_workers(root / name, available, root / output, control=control)
        if stage == "merge":
            return root / "corpus"
        if (root / 'artifact-cache').exists():
            shutil.rmtree(root / 'artifact-cache')
        options = dict(config.get("training", {}))
        options.setdefault("device", config.get("device", "cpu"))
        options.setdefault("backend", "jax" if options["device"] == "tpu" else "torch")
        if options.get("steps") is not None and options.get("epochs") is not None:
            raise ValueError("Choose training.steps or training.epochs")
        if options.get("steps") is None and options.get("epochs") is None:
            options["epochs"] = 1
        options.setdefault("steps", None)
        if config.get("feature_mode", "tokens") == "tokens":
            options.setdefault("online_target", config["model"])
            options.setdefault("teacher_engine_options", config.get("engine_options"))
        if (root / "validation-corpus").exists():
            options.setdefault("validation_corpus", root / "validation-corpus")
        if (root / "training/checkpoint.pt").exists():
            options.setdefault("resume", root / "training/checkpoint.pt")
        check(control, 'training')
        train(root / "corpus", root / "training", control=control, **options)
        return root / "training/draft"
    raise ValueError("stage must be all, worker, merge or train")


def run_job(config_path, *, stage='all', worker=0, artifacts=None, max_run_seconds=None,
            save_margin_seconds=None, checkpoint_interval_seconds=None, resume_from=None, _control=None):
    config = read_json(config_path) if isinstance(config_path, (str, Path)) else dict(config_path)
    runtime = dict(config.get('runtime', {}))
    unknown = set(runtime) - {'max_run_seconds', 'save_margin_seconds', 'checkpoint_interval_seconds',
                              'resume_from', 'archive_on_pause'}
    if unknown:
        raise ValueError(f'Unknown runtime options: {sorted(unknown)}')
    for key, value in dict(max_run_seconds=max_run_seconds, save_margin_seconds=save_margin_seconds,
                           checkpoint_interval_seconds=checkpoint_interval_seconds,
                           resume_from=resume_from).items():
        if value is not None:
            runtime[key] = value
    control = _control or RunControl(**{key: runtime[key] for key in
        ('max_run_seconds', 'save_margin_seconds', 'checkpoint_interval_seconds') if key in runtime})
    root = Path(config['out'])
    with control:
        restoring = False
        try:
            if runtime.get('resume_from'):
                restoring = True
                restore_job(runtime['resume_from'], root, control)
                restoring = False
            root.mkdir(parents=True, exist_ok=True)
            origin = root / 'model-origin.json'
            if not origin.exists():
                previous = read_json(root / 'job-config.json') if (root / 'job-config.json').exists() else config
                write_json(origin, dict(config_identity_path=str(previous['model'])))
            config['engine_options'] = {**(config.get('engine_options') or {}), **read_json(origin)}
            write_json(root / 'job-config.json', config)
            write_json(root / 'session.json', dict(status='running', requested_stage=stage, worker=worker))
            result = _run_job(config, stage=stage, worker=worker, artifacts=artifacts, control=control)
            write_json(root / 'session.json', dict(status='complete', requested_stage=stage,
                worker=worker, elapsed_seconds=control.elapsed(), result=str(result)))
            return result
        except SessionPaused as exc:
            root.mkdir(parents=True, exist_ok=True)
            status = dict(status='paused', requested_stage=stage, worker=worker,
                          paused_stage=exc.stage, reason=exc.reason, elapsed_seconds=control.elapsed())
            write_json(root / 'session.json', status)
            # Reserve time for this final write; no TPU computation is required.
            if runtime.get('archive_on_pause', True) and not restoring:
                archive = root.with_name(root.name + f'.resume-{stage}-{worker}.zip')
                try:
                    snapshot_job(root, archive)
                    status['resume_archive'] = str(archive)
                except OSError as error:
                    status['archive_error'] = str(error)
                    print(f'ZIP save failed: {error}. Preserve the full resume directory: {root}', flush=True)
                write_json(root / 'session.json', status)
            print(f"Paused at {exc.stage}. Resume directory: {root}", flush=True)
            if status.get('resume_archive'):
                print(f"Resume archive: {status['resume_archive']}", flush=True)
            return root
