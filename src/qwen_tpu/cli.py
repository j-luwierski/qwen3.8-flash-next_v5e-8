import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch

from .engine import load_engine, make_tiny
from .io import atomic_write
from .shards import capture_worker, make_plan, merge_workers
from .training import train


def layers(value):
    return [int(v) for v in value.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Qwen Flash-Next / SpecForge DFlash2 training pipeline")
    parser.add_argument("--threads", type=int, default=2)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("tiny-model", help="Create reduced Qwen4Exp with random weights for tests")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("demo-data")
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=8)
    p = sub.add_parser("tokenize", help="Tokenize complete text sequences, with loss on every token")
    p.add_argument("--data", required=True, help='JSONL with {"text": "complete training sequence"}')
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--tokenizer-identity", required=True, help="e.g. Qwen/Qwen3.8-Flash-Next@<commit>")
    p = sub.add_parser("capture")
    p.add_argument("--plan", required=True)
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--layers", type=layers, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--device", default="cpu")
    p.add_argument("--backend", choices=["torch", "jax"], default="torch")
    p.add_argument("--devices", default="all", help="Capture on all local devices (default) or the first N")
    p = sub.add_parser("merge")
    p.add_argument("--plan", required=True)
    p.add_argument("--workers", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("train")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    duration = p.add_mutually_exclusive_group()
    duration.add_argument("--steps", type=int)
    duration.add_argument("--epochs", type=int)
    p.add_argument("--mask-token-id", type=int, required=True)
    p.add_argument("--draft-layers", type=int, default=2)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=6e-4)
    p.add_argument("--anchors", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume")
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--device", choices=["cpu", "cuda", "tpu"], default="cpu")
    p.add_argument("--backend", choices=["torch", "jax"], default="torch")
    p.add_argument("--devices", default="all")
    p.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--objective-chunk-blocks", type=int, default=32)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--optimizer-offload", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--online-target")
    p.add_argument("--validation-corpus")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--sequence-length", type=int)
    p = sub.add_parser("generate", help="Greedy CPU generation, optionally verified DFlash2 proposals")
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--layers", type=layers, required=True)
    p.add_argument("--draft")
    p.add_argument("--input-ids", type=layers, required=True)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--backend", choices=["torch", "jax"], default="torch")
    p.add_argument("--device", choices=["cpu", "tpu"], default="cpu")
    p.add_argument("--devices", default="all", help="Distribute routed MoE experts across all local devices or the first N")
    p.add_argument("--temperature", type=float, default=0.)
    p.add_argument("--top-p", type=float, default=1.)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("fetch-prompts", help="Stream public HF prompts or normalize local conversations")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset")
    source.add_argument("--data")
    p.add_argument("--revision")
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--split", default="train")
    p.add_argument("--dataset-config")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p = sub.add_parser("run", help="Execute a resumable prompt-to-drafter job")
    p.add_argument("--config", required=True)
    p.add_argument("--stage", choices=["worker", "merge", "train", "all"], default="worker")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--artifacts", nargs="*")
    p = sub.add_parser("pack-worker")
    p.add_argument("--directory", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("prepare-weights", help="Stream 4/8-bit weight conversion, resumable on CPU")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bits", choices=[4, 8], type=int, default=8)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--chunk-mib", type=int, default=32)
    for name in ('run', 'capture', 'merge', 'train', 'fetch-prompts', 'prepare-weights'):
        timed = sub.choices[name]
        timed.add_argument('--max-run-seconds', type=float, help='Per-invocation wall clock budget')
        timed.add_argument('--save-margin-seconds', type=float, help='Reserve for checkpoint/archive writes')
        timed.add_argument('--checkpoint-interval-seconds', type=float, help='Periodic progress/checkpoint interval')
    sub.choices['run'].add_argument('--resume-from', help='Saved job directory or resume ZIP')
    args = vars(parser.parse_args())
    threads, command = args.pop("threads"), args.pop("command")
    if args.get("backend") == "jax" and args.get("device", "cpu") == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(threads)
    from .runtime import RunControl, SessionPaused
    from .io import write_json
    runtime = {key: args.pop(key) for key in ('max_run_seconds', 'save_margin_seconds', 'checkpoint_interval_seconds')
               if key in args and args[key] is not None}
    for key in ('max_run_seconds', 'save_margin_seconds', 'checkpoint_interval_seconds'):
        args.pop(key, None)
    control = RunControl(**runtime) if runtime and command != 'run' else None
    if command == 'run':
        args.update(runtime)
    elif control:
        args['control'] = control
    try:
        with control if control else nullcontext():
            if command == "prepare-weights":
                from .quantization import prepare_weights
                print(prepare_weights(**args))
            elif command == "run":
                from .workflow import run_job
                config = json.loads(Path(args["config"]).read_text())
                if config.get("device", "cpu") == "cpu":
                    os.environ.setdefault("JAX_PLATFORMS", "cpu")
                print(run_job(args.pop("config"), **args))
            elif command == "fetch-prompts":
                from .data import fetch_prompts
                print("samples", len(fetch_prompts(**args)))
            elif command == "pack-worker":
                from .workflow import pack_worker
                print(pack_worker(**args))
            elif command == "tiny-model":
                print(make_tiny(args["out"], args["seed"]))
            elif command == "demo-data":
                records = []
                for i in range(args["samples"]):
                    ids = [(j * 7 + i * 11) % 120 + 3 for j in range(24 + i % 3)]
                    records.append({"id": f"demo-{i}", "input_ids": ids, "loss_mask": [0] * 4 + [1] * (len(ids) - 4)})
                atomic_write(args["out"], lambda f: f.write("".join(json.dumps(r) + "\n" for r in records).encode()))
            elif command == "tokenize":
                from transformers import AutoTokenizer
                if len(args["revision"]) != 40:
                    parser.error("--revision must be an immutable 40-character commit")
                tokenizer = AutoTokenizer.from_pretrained(args["tokenizer"], revision=args["revision"], trust_remote_code=False)
                records = []
                for i, line in enumerate(Path(args["data"]).read_text().splitlines()):
                    raw = json.loads(line)
                    ids = tokenizer.encode(raw["text"], add_special_tokens=False)
                    records.append({"id": raw.get("id", str(i)), "input_ids": ids, "loss_mask": [1] * len(ids)})
                atomic_write(args["out"], lambda f: f.write("".join(json.dumps(r) + "\n" for r in records).encode()))
            elif command == "plan":
                print(make_plan(**args)["plan_id"])
            elif command == "capture":
                args["plan_dir"] = args.pop("plan")
                args["model_path"] = args.pop("model")
                print("samples", len(capture_worker(**args)["items"]))
            elif command == "merge":
                print("samples", len(merge_workers(args["plan"], args["workers"], args["out"], control=args.get("control"))["items"]))
            elif command == "train":
                train(**args)
            elif command == "generate":
                engine = load_engine(args["model"], args["layers"], revision=args["revision"], backend=args["backend"],
                                     device=args["device"], devices=args["devices"])
                try:
                    if args["draft"]:
                        from .speculative import greedy_speculate
                        ids, accepted = greedy_speculate(engine, args["draft"], args["input_ids"], args["max_new_tokens"])
                        print(json.dumps({"input_ids": ids, "accepted_lengths": accepted}))
                    else:
                        print(json.dumps({"input_ids": engine.generate(args["input_ids"], args["max_new_tokens"],
                            temperature=args["temperature"], top_p=args["top_p"], top_k=args["top_k"], seed=args["seed"])}))
                finally:
                    engine.close()

    except SessionPaused as exc:
        out = Path(args['out'])
        write_json(out.with_name(out.name + '.session.json'), dict(status='paused', stage=exc.stage, reason=exc.reason))
        print(str(exc))


if __name__ == "__main__":
    main()
