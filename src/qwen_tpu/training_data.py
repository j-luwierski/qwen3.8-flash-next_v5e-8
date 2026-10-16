"""Verified offline features or compact tokens with on-demand teacher capture."""

from pathlib import Path

import torch

from .engine import load_engine
from .io import read_json, sha256
from .shards import validate_features


class TrainingCorpus:
    def __init__(self, path, *, online_target=None, target_device="cpu", target_devices="all", engine_options=None):
        self.path = Path(path)
        self.manifest = read_json(self.path / "manifest.json")
        self.items = self.manifest["items"]
        self.contract = self.manifest["contract"]
        self.mode = self.manifest.get("feature_mode", "hidden_states")
        self.verified = set()
        self.teacher = None
        if self.mode == "tokens":
            if not online_target:
                raise ValueError("Token-only corpus requires --online-target with the matching target checkpoint")
            backend = "jax" if self.contract["backend"].startswith("jax") else "torch"
            options = (dict(engine_options or {}) if backend == "jax" else
                       {k: v for k, v in (engine_options or {}).items() if k == 'config_identity_path'})
            if (self.path / 'assets/origin.json').exists():
                options.update(read_json(self.path / 'assets/origin.json'))
            if backend == "jax":
                expected = "jax" if self.contract.get("qsa_selection") else "reference"
                if options.get("qsa_backend", expected) not in ("auto", expected):
                    raise ValueError("Requested QSA selection differs from the training corpus")
                options["qsa_backend"] = expected
            self.teacher = load_engine(online_target, self.contract["layer_ids"], backend=backend,
                                       device=target_device, devices=target_devices, dtype=self.contract["dtype"], **options)
            if any(self.contract.get(key) != value for key, value in self.teacher.contract.items()):
                self.teacher.close()
                raise ValueError("Online teacher does not match the corpus target contract")

    def _read_sample(self, index):
        item = self.items[index]
        path = self.path / "features" / f"{item['key']}.ckpt"
        if item["key"] not in self.verified:
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Corrupted training sample: {path}")
            self.verified.add(item["key"])
        raw = torch.load(path, map_location="cpu", weights_only=True)
        validate_features(raw, self.contract, item["length"], self.mode)
        return raw

    def sample(self, index):
        raw = self._read_sample(index)
        if self.teacher is not None:
            raw = self.teacher.capture(raw["input_ids"].tolist(), raw["loss_mask"].tolist())
        return raw

    def check_disjoint(self, other):
        if {item["key"] for item in self.items} & {item["key"] for item in other.items}:
            raise ValueError("Training and validation corpora contain overlapping samples")

    def batch(self, indices, *, padded_length=None):
        samples = [self.sample(i) for i in indices]
        length = max(s["input_ids"].numel() for s in samples)
        if padded_length is not None:
            if length > padded_length:
                raise ValueError("Training sequence exceeds configured length; data is never silently truncated")
            length = padded_length
        ids = torch.zeros(len(samples), length, dtype=torch.long)
        mask = torch.zeros(len(samples), length)
        hidden = torch.zeros(len(samples), length, self.contract["hidden_size"] * len(self.contract["layer_ids"]))
        for i, raw in enumerate(samples):
            size = raw["input_ids"].numel()
            ids[i, :size], mask[i, :size] = raw["input_ids"], raw["loss_mask"]
            hidden[i, :size] = raw["hidden_states"][0].float()
        return {"input_ids": ids, "hidden_states": hidden, "loss_mask": mask}

    def batch_jax(self, indices, *, padded_length, sharding):
        import jax
        import jax.numpy as jnp
        import numpy as np
        if self.teacher is None or not hasattr(self.teacher, "capture_device"):
            return {k: jax.device_put(v.numpy().astype(np.int32 if k == "input_ids" else np.float32), sharding)
                    for k, v in self.batch(indices, padded_length=padded_length).items()}
        collected = {k: [] for k in ("input_ids", "hidden_states", "loss_mask")}
        for index in indices:
            raw = self._read_sample(index)
            length = raw["input_ids"].numel()
            if length > padded_length:
                raise ValueError("Training sequence exceeds configured length")
            features = self.teacher.capture_device(raw["input_ids"].tolist(), raw["loss_mask"].tolist())
            for key in collected:
                value = features[key][0] if key == "hidden_states" else features[key]
                # First transfer directly to training devices, then pad there.
                value = jax.device_put(value, sharding)
                collected[key].append(jnp.pad(value, ((0, padded_length-length), *[(0, 0)] * (value.ndim-1))))
        return {k: jnp.stack(values) for k, values in collected.items()}

    def close(self):
        if self.teacher is not None:
            self.teacher.close()
