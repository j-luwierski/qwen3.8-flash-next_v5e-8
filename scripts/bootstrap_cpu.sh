#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SPEC_REV=d80daf31aacfd2193283ae36f36864e4dde199bc
TORCHAX_REV=d95b0a5bff13e2b945e6a0e7ba3b3937b838161b
uv venv --python 3.12 --allow-existing .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu'
uv pip install --python .venv/bin/python -e '.[test,jax,data]'
if [ ! -d third_party/SpecForge/.git ]; then
  git clone https://github.com/sgl-project/SpecForge.git third_party/SpecForge
fi
git -C third_party/SpecForge checkout --detach "$SPEC_REV"
# Upstream's full dependency set installs SGLang/CUDA. The reference CPU path
# imports its real architecture and objective directly, without those services.
uv pip install --python .venv/bin/python --no-deps -e third_party/SpecForge
if [ ! -f third_party/Torchax/SOURCE_REVISION ] && [ ! -d third_party/Torchax/.git ]; then
  git clone https://github.com/google/torchax.git third_party/Torchax
fi
if [ -d third_party/Torchax/.git ]; then
  git -C third_party/Torchax checkout --detach "$TORCHAX_REV"
fi
echo "$TORCHAX_REV" > third_party/Torchax/SOURCE_REVISION
uv pip install --python .venv/bin/python --no-deps -e third_party/Torchax
