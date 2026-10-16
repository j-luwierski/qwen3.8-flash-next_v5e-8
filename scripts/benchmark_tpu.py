"""Run on CPU first; TPU must be explicitly selected. No model downloads."""
import argparse
import json
import time
from pathlib import Path

import torch

from qwen_tpu.engine import load_engine
from qwen_tpu.io import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', choices=('cpu', 'tpu'), default='cpu')
    parser.add_argument('--devices', default='all')
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--new-tokens', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cache-mib', type=int, default=1024)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.device == 'cpu':
        import os
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    torch.set_num_threads(2)
    reports = []
    for mode, cache in [('reference', 0), ('jax', 0), ('jax', args.cache_mib)]:
        with load_engine(args.model, [0], backend='jax', device=args.device,
                         devices=args.devices, qsa_backend=mode, weight_cache_mib=cache) as engine:
            ids = [10 + i % min(100, engine.config.vocab_size - 10) for i in range(args.tokens)]
            start = time.perf_counter()
            engine.capture(ids, [1] * len(ids))
            cold = time.perf_counter() - start
            engine.generate(ids[:8], args.new_tokens, stop_token_ids=[])
            before = engine.weight_cache.stats()
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                engine.capture(ids, [1] * len(ids))
                times.append(time.perf_counter() - start)
            start = time.perf_counter()
            engine.generate(ids[:8], args.new_tokens, stop_token_ids=[])
            generation = time.perf_counter() - start
            after = engine.weight_cache.stats()
            reports.append(dict(qsa_backend=mode, cache_mib=cache, cold_prefill_seconds=cold,
                warm_prefill_seconds=times, generated_tokens=args.new_tokens,
                generation_seconds=generation, generation_tokens_per_second=args.new_tokens/generation,
                weight_upload_bytes_during_measurement=after['host_to_device_weight_bytes']-before['host_to_device_weight_bytes'],
                cache=after, contract=engine.contract))
    write_json(args.out, dict(platform=args.device, devices=args.devices, tokens=args.tokens, results=reports,
                              note='Different QSA tie policies; throughput comparison is not a quality evaluation.'))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
