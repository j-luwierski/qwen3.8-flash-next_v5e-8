"""Content identities and crash-safe local writes. No network side effects."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, lambda f: f.write((json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()))


def read_json(path):
    return json.loads(Path(path).read_text())


def checkpoint_identity(directory):
    """Hash all referenced shards, including user-created links to mounted data."""
    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    mapping = read_json(index)["weight_map"] if index.exists() else None
    names = sorted(set(mapping.values())) if mapping else sorted(p.name for p in directory.glob("*.safetensors"))
    if not names:
        raise ValueError("Local model must contain safetensors weights")
    for name in names:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Invalid checkpoint shard path")
    payload = {"weight_map": mapping, "shards": {name: sha256(directory / name) for name in names}}
    if (directory / "qwen_quantization.json").exists():
        payload["quantization"] = sha256(directory / "qwen_quantization.json")
    return digest(payload)
