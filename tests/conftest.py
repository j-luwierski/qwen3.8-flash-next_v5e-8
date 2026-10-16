import os

os.environ["JAX_PLATFORMS"] = "cpu"

import pytest
import torch

from qwen_tpu.engine import make_tiny
from qwen_tpu.io import atomic_write


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture(scope="session")
def target(tmp_path_factory):
    return make_tiny(tmp_path_factory.mktemp("models") / "target")


@pytest.fixture
def data(tmp_path):
    import json
    path = tmp_path / "data.jsonl"
    records = [{"id": f"sample-{i}", "input_ids": [(i * 13 + j * 7) % 120 + 3 for j in range(16 + i % 2)],
                "loss_mask": [0, 0] + [1] * (14 + i % 2)} for i in range(4)]
    atomic_write(path, lambda f: f.write("".join(json.dumps(r) + "\n" for r in records).encode()))
    return path
