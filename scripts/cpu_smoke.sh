#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
run_dir=${1:-$(mktemp -d runs/cpu-smoke-XXXXXX)}
py=.venv/bin/python
export JAX_PLATFORMS=cpu
"$py" -m qwen_tpu.cli tiny-model --out "$run_dir/target"
"$py" - "$run_dir" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
config=json.loads(Path('configs/job.cpu.json').read_text())
config.update(model=str(root/'target'),out=str(root/'job'))
(root/'job.json').write_text(json.dumps(config,indent=2)+'\n')
PY
"$py" -m qwen_tpu.cli run --config "$run_dir/job.json" --stage worker --worker 0 &
worker_a=$!
"$py" -m qwen_tpu.cli run --config "$run_dir/job.json" --stage worker --worker 1 &
worker_b=$!
wait "$worker_a"
wait "$worker_b"
"$py" -m qwen_tpu.cli run --config "$run_dir/job.json" --stage train \
  --artifacts "$run_dir/job/worker-0.zip" "$run_dir/job/worker-1.zip"
echo "CPU job complete: $run_dir/job/training/draft"
