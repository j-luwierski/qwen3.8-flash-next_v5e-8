"""Streaming text-only Flash-Next prefill in native JAX.

Equations and tensor layout adapted from Hugging Face Transformers Qwen4Exp
(Qwen Team / Hugging Face, Apache-2.0; see third_party/transformers.LICENSE).

Correctness path, not the optimized GLM serving implementation. Parameters are
read from local BF16/F32 HF safetensors on demand. PLE tables stay on the host;
only requested embedding rows and active experts are copied to the device.
No entire model, layer expert bank, or PLE table has to fit accelerator memory.
QSA gathers selected keys in query tiles without a dense sequence-square mask.
"""

import re
from functools import wraps
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig

from .engine import FEATURE_POLICY, config_identity
from .io import checkpoint_identity, digest, read_json, write_json


def on_selected_device(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        # Also place temporary masks/aranges on the selected device when a
        # machine has both a CPU and another JAX backend installed.
        with self.jax.default_device(self.device):
            return method(self, *args, **kwargs)
    return call


class TensorReader:
    def __init__(self, directory):
        directory = Path(directory)
        index = directory / "model.safetensors.index.json"
        self.stack = ExitStack()
        self.files = {}
        self.paths = {}
        self.mapping = {}
        quantized = directory / "qwen_quantization.json"
        self.quantized = read_json(quantized)["tensors"] if quantized.exists() else None
        self.directory = directory
        if self.quantized:
            self.mapping = {key.replace("model.language_model.", "model."): key for key in self.quantized}
            return
        if index.exists():
            mapping = read_json(index)["weight_map"]
            for key, shard in mapping.items():
                if Path(shard).is_absolute() or ".." in Path(shard).parts:
                    raise ValueError("Checkpoint shard escapes model directory")
                # Links deliberately prepared by the user may point to shards
                # mounted from several read-only Kaggle input datasets.
                path = directory / shard
                self.paths[key] = path
        else:
            for path in sorted(directory.glob("*.safetensors")):
                with safe_open(path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        self.paths[key] = path
        if not self.paths:
            raise ValueError("No local HF safetensors checkpoint found")
        for key in self.paths:
            self.mapping[key.replace("model.language_model.", "model.")] = key

    def slice(self, key):
        key = self.mapping[key]
        if self.quantized is not None:
            from .quantization import QuantizedSlice
            return QuantizedSlice(self, self.quantized[key])
        path = self.paths[key]
        if path not in self.files:
            self.files[path] = self.stack.enter_context(safe_open(path, framework="pt", device="cpu"))
        view = self.files[path].get_slice(key)
        if view.get_dtype() not in ("BF16", "F32", "F16", "I64", "I32"):
            raise ValueError(f"Unsupported checkpoint dtype for {key}: {view.get_dtype()}; use BF16 weights")
        return view

    def _file(self, name):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Quantized shard escapes model directory")
        path = self.directory / name
        if path not in self.files:
            if len(self.files) >= 32:
                oldest = next(iter(self.files))
                self.files.pop(oldest).__exit__(None, None, None)
            handle = safe_open(path, framework="pt", device="cpu")
            handle.__enter__()
            self.files[path] = handle
        return self.files[path]

    def tensor(self, key, selection=None):
        value = self.slice(key)
        result = value[:] if selection is None else value[selection]
        return result.float().numpy() if result.is_floating_point() else result.numpy()

    def rows(self, key, indices):
        shape = np.shape(indices)
        unique, inverse = np.unique(np.asarray(indices).reshape(-1), return_inverse=True)
        view = self.slice(key)
        rows = np.concatenate([view[int(i):int(i) + 1].float().numpy() for i in unique], axis=0)
        return rows[inverse].reshape(*shape, rows.shape[-1])

    def close(self):
        if self.quantized is not None:
            for handle in self.files.values():
                handle.__exit__(None, None, None)
        self.stack.close()
        self.files.clear()


class JaxQwenEngine:
    def __init__(self, model_path, layer_ids, *, revision=None, dtype="float32", device="cpu", max_tokens=None,
                 device_index=0, _weight_identity=None, query_chunk_size=None,
                 qsa_backend="auto", weight_cache_mib=None, kv_cache_block_size=256,
                 compilation_cache_dir=None, expert_token_tile=32, lm_head_chunk_size=16384,
                 config_identity_path=None):
        import jax
        import jax.numpy as jnp

        self.jax, self.jnp = jax, jnp
        if device not in ("cpu", "tpu") or dtype not in ("float32", "bfloat16"):
            raise ValueError("JAX backend supports cpu/tpu and float32/bfloat16")
        if not Path(model_path).is_dir():
            raise ValueError("Streaming JAX requires a local checkpoint directory; attach HF shards as a dataset")
        available = jax.local_devices(backend=device)
        if type(device_index) is not int or not 0 <= device_index < len(available):
            raise ValueError(f"device_index must be in 0..{len(available) - 1}")
        self.device = available[device_index]
        self.dtype = getattr(jnp, dtype)
        if qsa_backend not in ("auto", "reference", "jax"):
            raise ValueError("qsa_backend must be auto, reference or jax")
        self.qsa_backend = ("jax" if device == "tpu" else "reference") if qsa_backend == "auto" else qsa_backend
        self.optimized = self.qsa_backend == "jax"
        query_chunk_size = query_chunk_size or (128 if self.optimized else 16)
        if query_chunk_size < 1:
            raise ValueError("query_chunk_size must be positive")
        self.query_chunk_size = query_chunk_size
        if min(kv_cache_block_size, expert_token_tile, lm_head_chunk_size) < 1:
            raise ValueError("Cache block, expert tile and LM head chunk must be positive")
        self.kv_cache_block_size = kv_cache_block_size
        self.expert_token_tile = expert_token_tile
        self.lm_head_chunk_size = lm_head_chunk_size
        if compilation_cache_dir:
            jax.config.update("jax_compilation_cache_dir", str(Path(compilation_cache_dir).resolve()))
        from .device_cache import DeviceWeightCache
        from .jax_kernels import device_kernels
        self.weight_cache = DeviceWeightCache((1024 if device == "tpu" else 0) if weight_cache_mib is None else weight_cache_mib)
        self._linear_kernel, self._norm_kernel, self._expert_kernel, self._update_kv = device_kernels()
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=False).get_text_config()
        c = self.config
        self.config_identity_path = (config_identity_path if config_identity_path is not None
                                     else c.to_dict().get('_name_or_path', ''))
        self.max_tokens = min(max_tokens or c.max_position_embeddings, c.max_position_embeddings)
        if c.model_type != "qwen4_exp_text" or c.rope_parameters["rope_type"] != "default":
            raise ValueError("JAX currently supports text Qwen4Exp with default RoPE only")
        if c.attention_bias or c.hidden_act != "silu":
            raise ValueError("JAX reference requires bias-free projections and SiLU")
        self.layer_ids = list(layer_ids)
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids) or any(
                i < 0 or i >= c.num_hidden_layers for i in self.layer_ids):
            raise ValueError("Capture layer ids must be unique zero-based decoder indices")
        self.contract = {
            "model_type": "qwen4_exp_text", "weights": _weight_identity or checkpoint_identity(model_path),
            "config": config_identity(c, config_identity_path), "dtype": dtype,
            "backend": "jax-streaming-v1", "feature_policy": FEATURE_POLICY,
            "layer_ids": self.layer_ids, "hidden_size": c.hidden_size, "vocab_size": c.vocab_size,
        }
        if self.optimized:
            # JAX has deterministic lower-index tie resolution. PyTorch's CPU
            # topk chooses other equally scored blocks: never mix these corpora.
            self.contract.update(backend="jax-streaming-v2", qsa_selection="jax.top_k.lower_index.v1")
        self.reader = TensorReader(model_path)
        self._delta_scan = jax.jit(self._scan_delta)

    def _put(self, x):
        if isinstance(x, self.jax.Array):
            return self.jax.device_put(x.astype(self.dtype), self.device)
        # BF16 on the host before DMA: do not transfer a whole FP32 matrix and
        # only then convert it in scarce device memory.
        return self.jax.device_put(np.asarray(x, dtype=np.dtype(self.dtype)), self.device)

    def _weight(self, name, selection=None):
        key = (name, (selection.start, selection.stop) if isinstance(selection, slice) else selection)
        cached = self.weight_cache.get(key)
        if cached is not None:
            return cached
        return self.weight_cache.put(key, self._put(self.reader.tensor(name, selection)))

    def _lin(self, x, name):
        return self._linear_kernel(x, self._weight(name + ".weight"))

    def _norm(self, x, name, group=None, offset=True):
        jnp = self.jnp
        shape, dtype = x.shape, x.dtype
        weight = self._weight(name + ".weight")
        if self.optimized:
            if group:
                grouped = x.reshape(*shape[:-1], -1, group)
                return self._norm_kernel(grouped, weight.reshape(-1, group),
                                         self.config.rms_norm_eps, int(offset)).reshape(shape)
            return self._norm_kernel(x, weight, self.config.rms_norm_eps, int(offset))
        x = x.astype(jnp.float32)
        if group:
            x = x.reshape(*shape[:-1], -1, group)
        x = x * self.jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.config.rms_norm_eps)
        x = x.reshape(shape)
        weight = weight.astype(jnp.float32)
        return (x * (weight + (1 if offset else 0))).astype(dtype)

    def _hc(self, streams, prefix, inject=True):
        c, jnp = self.config, self.jnp
        if self.optimized:
            from .jax_kernels import hyper_connection_kernel
            return hyper_connection_kernel(c.hc_count, c.hidden_size, c.rms_norm_eps, inject)(
                streams, self._weight(prefix + ".hc_norm.weight"),
                self._weight(prefix + ".input_mix_weight_down.weight"),
                self._weight(prefix + ".input_mix_weight_up.weight"),
                self._weight(prefix + ".block_inject_weight.weight") if inject else jnp.empty((0, 0), streams.dtype))
        normed = self._norm(streams, prefix + ".hc_norm", c.hidden_size)
        weights = self.jax.nn.sigmoid(self._lin(self.jax.nn.silu(
            self._lin(normed, prefix + ".input_mix_weight_down") / c.hc_count),
            prefix + ".input_mix_weight_up"))
        mixed = (weights * normed).reshape(*streams.shape[:-1], c.hc_count, c.hidden_size).mean(axis=-2)
        if not inject:
            return mixed
        injection = 2 * self.jax.nn.sigmoid(self._lin(normed, prefix + ".block_inject_weight") / c.hc_count)
        return mixed, injection

    def _conv(self, x, name, dilation=1, cache=None):
        weights = self._weight(name + ".weight")[:, 0]
        taps, length = weights.shape[-1], x.shape[0]
        history_size = dilation * (taps - 1)
        history = cache.get(name) if cache is not None else None
        if history is None:
            history = self.jnp.zeros((history_size, x.shape[-1]), dtype=x.dtype)
        padded = self.jnp.concatenate([history, x], axis=0)
        if cache is not None:
            cache[name] = padded[-history_size:] if history_size else padded[:0]
        if self.optimized:
            from .jax_kernels import convolution_kernel
            return convolution_kernel(dilation)(padded, weights)
        out = self.jnp.zeros_like(x)
        for tap in range(taps):
            start = tap * dilation
            out = out + padded[start:start + length] * weights[:, tap]
        return self.jax.nn.silu(out)

    def _scan_delta(self, q, k, v, decay, beta, initial):
        jnp = self.jnp
        q = q.astype(jnp.float32)
        k = k.astype(jnp.float32)
        q = q * self.jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6) / q.shape[-1] ** 0.5
        k = k * self.jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)

        def step(state, row):
            qt, kt, vt, gt, bt = row
            state = state * jnp.exp(gt)[:, None, None]
            delta = (vt - jnp.einsum("hk,hkv->hv", kt, state)) * bt[:, None]
            state = state + kt[:, :, None] * delta[:, None, :]
            return state, jnp.einsum("hk,hkv->hv", qt, state)

        state, output = self.jax.lax.scan(step, initial, (q, k, v.astype(jnp.float32), decay, beta))
        return state, output.astype(v.dtype)

    def _delta(self, x, prefix, cache=None):
        c, jnp = self.config, self.jnp
        size = c.linear_num_key_heads * c.linear_key_head_dim
        qkv = self._conv(self._lin(x, prefix + ".in_proj_qkv"), prefix + ".conv1d", cache=cache)
        q, k, v = jnp.split(qkv, [size, size * 2], axis=-1)
        q = q.reshape(-1, c.linear_num_key_heads, c.linear_key_head_dim)
        k = k.reshape(q.shape)
        v = v.reshape(-1, c.linear_num_value_heads, c.linear_value_head_dim)
        repeats = c.linear_num_value_heads // c.linear_num_key_heads
        q, k = jnp.repeat(q, repeats, axis=1), jnp.repeat(k, repeats, axis=1)
        beta = self.jax.nn.sigmoid(self._lin(x, prefix + ".in_proj_b"))
        a = self._lin(x, prefix + ".in_proj_a").astype(jnp.float32)
        g = -jnp.exp(self._weight(prefix + ".A_log").astype(jnp.float32)) * self.jax.nn.softplus(
            a + self._weight(prefix + ".dt_bias").astype(jnp.float32))
        initial = cache.get(prefix) if cache is not None else None
        if initial is None:
            initial = self._put(np.zeros((c.linear_num_value_heads, c.linear_key_head_dim,
                                          c.linear_value_head_dim), dtype=np.float32)).astype(jnp.float32)
        recurrent, core = self._delta_scan(q, k, v, g, beta, initial)
        if cache is not None:
            cache[prefix] = recurrent
        core = self._norm(core, prefix + ".norm", offset=False)
        z = self._lin(x, prefix + ".in_proj_z").reshape(core.shape).astype(jnp.float32)
        gate = self.jax.nn.sigmoid(z) if c.output_gate_type == "sigmoid" else self.jax.nn.silu(z)
        return self._lin((core * gate).astype(x.dtype).reshape(x.shape[0], -1), prefix + ".out_proj")

    def _rope(self, x, positions):
        c, jnp = self.config, self.jnp
        dim = int(c.head_dim * c.rope_parameters.get("partial_rotary_factor", 1.0))
        frequency = c.rope_parameters["rope_theta"] ** (-jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
        freq = jnp.asarray(positions)[:, None] * frequency[None, :]
        freq = jnp.concatenate([freq, freq], axis=-1)[:, None, :]
        rotary, rest = x[..., :dim], x[..., dim:]
        rotated = jnp.concatenate([-rotary[..., dim // 2:], rotary[..., :dim // 2]], axis=-1)
        return jnp.concatenate([rotary * jnp.cos(freq).astype(x.dtype) + rotated * jnp.sin(freq).astype(x.dtype), rest], axis=-1)

    def _attention(self, x, prefix, cache=None):
        c, jnp = self.config, self.jnp
        n = x.shape[0]
        old = cache.get(prefix) if cache is not None else None
        offset = 0 if old is None else (old["length"] if self.optimized else old[0].shape[0])
        positions = np.arange(offset, offset + n)
        qgate = self._lin(x, prefix + ".q_proj").reshape(n, c.num_attention_heads, c.head_dim * 2)
        q, gate = jnp.split(qgate, 2, axis=-1)
        k = self._lin(x, prefix + ".k_proj").reshape(n, c.num_key_value_heads, c.head_dim)
        v = self._lin(x, prefix + ".v_proj").reshape(k.shape)
        q = self._rope(self._norm(q, prefix + ".q_norm"), positions)
        k = self._rope(self._norm(k, prefix + ".k_norm"), positions)
        ip = prefix + ".indexer"
        index = self._lin(x, ip + ".index_qk_proj").reshape(n, c.indexer_n_heads + 1, c.indexer_head_dim)
        iq = self._rope(self._norm(index[:, :-1], ip + ".q_layernorm"), positions)
        raw = index[:, -1]
        if self.optimized:
            if cache is not None:
                total = offset + n
                capacity = min(self.max_tokens, ((total + self.kv_cache_block_size - 1) // self.kv_cache_block_size) * self.kv_cache_block_size)
                if old is None:
                    stored = (jnp.zeros((capacity, *k.shape[1:]), k.dtype),
                              jnp.zeros((capacity, *v.shape[1:]), v.dtype),
                              jnp.zeros((capacity, *raw.shape[1:]), raw.dtype))
                else:
                    stored = old["arrays"]
                    if stored[0].shape[0] < capacity:
                        stored = tuple(jnp.pad(a, ((0, capacity - a.shape[0]), *[(0, 0)] * (a.ndim - 1))) for a in stored)
                k, v, raw = self._update_kv(*stored, k, v, raw, jnp.int32(offset))
                cache[prefix] = {"arrays": (k, v, raw), "length": total}
            return self._optimized_attention(q, iq, k, v, raw, gate, positions, prefix)
        if old is not None:
            k, v, raw = (jnp.concatenate([past, new], axis=0) for past, new in zip(old, (k, v, raw)))
        if cache is not None:
            cache[prefix] = (k, v, raw)
        ratio, total = c.indexer_compress_ratio, k.shape[0]
        blocks = total // ratio
        pooled = None
        if blocks:
            pooled = raw[:blocks * ratio].reshape(blocks, ratio, -1).astype(jnp.float32).mean(axis=1).astype(x.dtype)
            pooled = self._rope(self._norm(pooled, ip + ".k_layernorm")[:, None], np.arange(blocks) * ratio)[:, 0]
        chunks = []
        groups = c.num_attention_heads // c.num_key_value_heads
        for begin in range(0, n, self.query_chunk_size):
            end = min(n, begin + self.query_chunk_size)
            width = end - begin
            # Score compressed blocks only for this query tile. Never allocate
            # [sequence, sequence] masks or full-token attention matrices.
            if blocks:
                scores = self.jax.nn.relu(jnp.einsum("qhd,bd->qbh", iq[begin:end].astype(jnp.float32),
                                                   pooled.astype(jnp.float32))).sum(-1) / c.indexer_head_dim ** 0.5
                host_scores = torch.from_numpy(np.asarray(scores).copy())
            selected = np.zeros((width, min(total, c.indexer_budget + ratio - 1)), dtype=np.int32)
            valid = np.zeros(selected.shape, dtype=bool)
            for row, position in enumerate(positions[begin:end]):
                complete = (int(position) + 1) // ratio
                chosen = np.empty(0, dtype=np.int32)
                if complete:
                    ids = host_scores[row, :complete].topk(min(complete, c.indexer_budget // ratio)).indices.numpy()
                    chosen = (ids[:, None] * ratio + np.arange(ratio)[None]).reshape(-1)
                tail = np.arange(complete * ratio, int(position) + 1)
                chosen = np.concatenate([chosen, tail]).astype(np.int32)
                selected[row, :len(chosen)], valid[row, :len(chosen)] = chosen, True
            selected = self.jax.device_put(selected, self.device)
            valid = self.jax.device_put(valid, self.device)
            keys, values = k[selected], v[selected]
            query = q[begin:end].reshape(width, c.num_key_value_heads, groups, c.head_dim)
            logits = jnp.einsum("qhgd,qkhd->qhgk", query, keys).astype(jnp.float32) / c.head_dim ** 0.5
            probs = self.jax.nn.softmax(jnp.where(valid[:, None, None, :], logits, -jnp.inf), axis=-1).astype(x.dtype)
            output = jnp.einsum("qhgk,qkhd->qhgd", probs, values).reshape(width, c.num_attention_heads, c.head_dim)
            chunks.append(output)
        output = jnp.concatenate(chunks, axis=0) * self.jax.nn.sigmoid(gate)
        return self._lin(output.reshape(n, -1), prefix + ".o_proj")

    def _optimized_attention(self, q, iq, k, v, raw, gate, positions, prefix):
        from .jax_kernels import sparse_qsa_kernel
        c, jnp = self.config, self.jnp
        n, ratio, tile = q.shape[0], c.indexer_compress_ratio, self.query_chunk_size
        blocks = k.shape[0] // ratio
        if blocks:
            pooled = raw[:blocks * ratio].reshape(blocks, ratio, -1).astype(jnp.float32).mean(1).astype(q.dtype)
            pooled = self._rope(self._norm(pooled, prefix + ".indexer.k_layernorm")[:, None],
                                jnp.arange(blocks) * ratio)[:, 0]
        else:
            pooled = jnp.empty((0, iq.shape[-1]), q.dtype)
        padded = ((n + tile - 1) // tile) * tile
        pad = lambda x: jnp.pad(x, ((0, padded - n), *[(0, 0)] * (x.ndim - 1)))
        output = sparse_qsa_kernel(ratio, c.indexer_budget, tile)(pad(q), pad(iq), k, v, pooled, pad(jnp.asarray(positions)))[:n]
        return self._lin((output * self.jax.nn.sigmoid(gate)).reshape(n, -1), prefix + ".o_proj")

    def _moe(self, x, prefix):
        c, jnp = self.config, self.jnp
        logits = self._lin(x, prefix + ".gate").astype(jnp.float32)
        probabilities = self.jax.nn.softmax(logits, axis=-1)
        weights, ids = self.jax.lax.top_k(probabilities, c.num_experts_per_tok)
        if c.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdims=True)
        ids = np.asarray(ids)
        dispatch = getattr(self, "_expert_dispatch", None)
        if dispatch is None:
            result = self._routed_experts(x, prefix, ids, weights, np.unique(ids))
        else:
            result = dispatch(x, prefix, ids, weights)
        sp = prefix + ".shared_expert"
        shared = self._lin(self.jax.nn.silu(self._lin(x, sp + ".gate_proj")) * self._lin(x, sp + ".up_proj"), sp + ".down_proj")
        return result + shared * self.jax.nn.sigmoid(self._lin(x, prefix + ".shared_expert_gate"))

    @on_selected_device
    def _routed_experts(self, x, prefix, ids, weights, experts):
        jnp = self.jnp
        x, weights = self._put(x), self._put(weights)
        result = jnp.zeros_like(x)
        # No [tokens, topk, hidden, intermediate] gather, and no whole expert bank.
        for expert in experts:
            token_idx, slot = np.where(ids == expert)
            if self.optimized:
                # Bucket dispatch shapes so expert kernels are reusable when
                # different experts receive different numbers of tokens.
                length = len(token_idx)
                padded = ((length + self.expert_token_tile - 1) // self.expert_token_tile) * self.expert_token_tile
                indices = self.jax.device_put(np.pad(token_idx.astype(np.int32), (0, padded - length)), self.device)
                slots = self.jax.device_put(np.pad(slot.astype(np.int32), (0, padded - length)), self.device)
                routing = weights[indices, slots] * (self.jnp.arange(padded) < length)
                if prefix + ".experts.gate_up_proj" in self.reader.mapping:
                    gu = self._weight(prefix + ".experts.gate_up_proj", int(expert))
                    down = self._weight(prefix + ".experts.down_proj", int(expert))
                else:
                    ep = prefix + f".experts.{expert}"
                    gu = jnp.concatenate((self._weight(ep + ".gate_proj.weight"), self._weight(ep + ".up_proj.weight")))
                    down = self._weight(ep + ".down_proj.weight")
                output = self._expert_kernel(x[indices], gu, down, routing)
                result = result.at[indices].add(output)
                result.block_until_ready()
                continue
            current = x[token_idx]
            if prefix + ".experts.gate_up_proj" in self.reader.mapping:
                gu = self._weight(prefix + ".experts.gate_up_proj", int(expert))
                down = self._weight(prefix + ".experts.down_proj", int(expert))
                gate, up = jnp.split(current @ gu.T, 2, axis=-1)
                output = (self.jax.nn.silu(gate) * up) @ down.T
            else:
                ep = prefix + f".experts.{expert}"
                output = self._lin(self.jax.nn.silu(self._lin(current, ep + ".gate_proj")) * self._lin(current, ep + ".up_proj"), ep + ".down_proj")
            result = result.at[token_idx].add(output * self._put(weights[token_idx, slot])[:, None])
            # Complete pending work before releasing this expert's device buffers.
            result.block_until_ready()
        return result

    def _ple(self, streams, ids, prefix, cache=None):
        c, jnp = self.config, self.jnp
        current_length = len(ids)
        if cache is not None:
            previous = cache.get(prefix + ".tokens", [])
            ids = np.concatenate([np.asarray(previous, dtype=np.int64), ids])
            cache[prefix + ".tokens"] = ids[-(c.ngram_size - 1):].tolist()
        ep = prefix + ".ple_embedding"
        multipliers = self.reader.tensor(ep + ".layer_multipliers").astype(np.uint64)
        vocab = self.reader.tensor(ep + ".ngram_heads_vocab_sizes")
        offsets = self.reader.tensor(ep + ".ngram_heads_offsets")
        eos = c.eos_token_id[0] if isinstance(c.eos_token_id, list) else c.eos_token_id
        shifted = [np.asarray(ids, dtype=np.uint64)]
        for shift in range(1, c.ngram_size):
            row, start = [], 0
            for i, token in enumerate(ids):
                row.append(ids[i - shift] if i - shift >= start else eos)
                if token == eos:
                    start = i + 1
            shifted.append(np.asarray(row, dtype=np.uint64))
        hashes = []
        for ngram in range(2, c.ngram_size + 1):
            mixed = shifted[0] * multipliers[0]
            for position in range(1, ngram):
                mixed = mixed ^ (shifted[position] * multipliers[position])
            begin, end = (ngram - 2) * c.heads_per_ngram, (ngram - 1) * c.heads_per_ngram
            hashes.append(mixed.view(np.int64)[:, None] % vocab[begin:end] + offsets[begin:end])
        indices = np.concatenate(hashes, axis=-1)[-current_length:]
        base = ep + ".ngram_embedding"
        if base + ".weight" in self.reader.mapping:
            embedding = self.reader.rows(base + ".weight", indices)
        else:
            names = sorted([k for k in self.reader.mapping if k.startswith(base + ".shard_")],
                           key=lambda k: int(re.search(r"shard_(\d+)", k).group(1)))
            embedding, offset = None, 0
            for name in names:
                size = self.reader.slice(name).get_shape()[0]
                selected = (indices >= offset) & (indices < offset + size)
                if selected.any():
                    rows = self.reader.rows(name, indices[selected] - offset)
                    if embedding is None:
                        embedding = np.empty((*indices.shape, rows.shape[-1]), dtype=np.float32)
                    embedding[selected] = rows
                offset += size
            if embedding is None or np.any(indices >= offset):
                raise ValueError("Incomplete PLE embedding shards")
        embedding = self._put(embedding.reshape(current_length, -1))
        key = self._norm(self._lin(embedding, prefix + ".key_proj"), prefix + ".norm_key", c.hidden_size)
        query = self._norm(streams, prefix + ".norm_query", c.hidden_size)
        gate = (key * query).reshape(current_length, c.hc_count, c.hidden_size).sum(-1, keepdims=True) / c.hidden_size ** 0.5
        gate = jnp.sign(gate) * jnp.sqrt(jnp.maximum(jnp.abs(gate), 1e-6))
        value = self._lin(embedding, prefix + ".value_proj")
        value = (self.jax.nn.sigmoid(gate) * value[:, None]).reshape(streams.shape)
        return value + self._conv(self._norm(value, prefix + ".norm_conv", c.hidden_size), prefix + ".conv1d", c.ngram_size, cache=cache)

    @on_selected_device
    def capture(self, input_ids, loss_mask, *, _cache=None, _collect=True, _device_features=False):
        ids, mask = np.asarray(input_ids), np.asarray(loss_mask, dtype=np.float32)
        if ids.ndim != 1 or len(ids) < 1 or mask.shape != ids.shape or len(ids) > self.max_tokens:
            raise ValueError(f"Expected matching token/mask vectors with length 1..{self.max_tokens}")
        if ids.dtype.kind not in "iu" or ids.min() < 0 or ids.max() >= self.config.vocab_size:
            raise ValueError("Invalid token ids")
        c, states = self.config, {}
        x = self._put(self.reader.rows("model.embed_tokens.weight", ids))
        streams = self.jnp.tile(x, (1, c.hc_count))
        for layer, ltype in enumerate(c.layer_types):
            p = f"model.layers.{layer}"
            if layer + 1 in c.ple_layer_ids:
                streams = streams + self._ple(streams, ids, p + ".ple", cache=_cache)
            hidden, injection = self._hc(streams, p + ".attn_hyper_connection")
            hidden = self._delta(hidden, p + ".linear_attn", cache=_cache) if ltype == "linear_attention" else self._attention(hidden, p + ".self_attn", cache=_cache)
            streams = streams + (injection[:, :, None] * hidden[:, None]).reshape(streams.shape)
            hidden, injection = self._hc(streams, p + ".mlp_hyper_connection")
            hidden = self._moe(hidden, p + ".mlp")
            streams = streams + (injection[:, :, None] * hidden[:, None]).reshape(streams.shape)
            if _collect and layer in self.layer_ids:
                value = self._hc(streams, "model.hyper_connection_mixer", inject=False)
                states[layer] = value if _device_features else np.asarray(value).copy()
            streams.block_until_ready()
        device_final = self._hc(streams, "model.hyper_connection_mixer", inject=False)
        if _device_features:
            return {"input_ids": self.jax.device_put(ids.astype(np.int32), self.device),
                    "loss_mask": self.jax.device_put(mask, self.device),
                    "hidden_states": self.jnp.concatenate([states[i] for i in self.layer_ids], -1)[None],
                    "target_last_hidden_states": device_final[None]}
        if not _collect and self.optimized:
            return device_final[None]
        final = np.asarray(device_final).copy()
        if not _collect:
            return torch.tensor(final.astype(np.float32))[None]
        features = {"input_ids": torch.tensor(ids, dtype=torch.long), "loss_mask": torch.tensor(mask),
                    "hidden_states": torch.tensor(np.concatenate([states[i] for i in self.layer_ids], axis=-1).astype(np.float32))[None],
                    "target_last_hidden_states": torch.tensor(final.astype(np.float32))[None]}
        if any(not torch.isfinite(v).all() for v in features.values()):
            raise ValueError("Non-finite JAX teacher features")
        return features

    def capture_device(self, input_ids, loss_mask):
        """Online JAX training: keep teacher features on the accelerator."""
        return self.capture(input_ids, loss_mask, _device_features=True)

    def embed(self, ids):
        return torch.tensor(self.reader.rows("model.embed_tokens.weight", np.asarray(ids)), dtype=torch.float32)

    @on_selected_device
    def logits(self, hidden):
        hidden = self._put(hidden.detach().float().numpy() if isinstance(hidden, torch.Tensor) else hidden)
        chunks = []
        chunk = self.lm_head_chunk_size if self.optimized else 1024
        for start in range(0, self.config.vocab_size, chunk):
            weight = self._weight("lm_head.weight", slice(start, start + chunk))
            value = self._linear_kernel(hidden, weight)
            value.block_until_ready()
            chunks.append(value if self.optimized else np.asarray(value).copy())
        if self.optimized:
            return torch.tensor(np.asarray(self.jnp.concatenate(chunks, axis=-1)).astype(np.float32))
        return torch.tensor(np.concatenate(chunks, axis=-1).astype(np.float32))

    def generate(self, input_ids, max_new_tokens, **options):
        from .generation import generate
        return generate(self, input_ids, max_new_tokens, **options)

    def prefill(self, input_ids):
        state = {}
        hidden = self.capture(input_ids, [1] * len(input_ids), _cache=state, _collect=False)
        state["length"] = len(input_ids)
        return state, hidden

    def decode(self, token, state):
        if state["length"] >= self.max_tokens:
            raise ValueError("Cached generation exceeds context limit")
        hidden = self.capture([token], [1], _cache=state, _collect=False)
        state["length"] += 1
        return state, hidden

    def save_assets(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tensors = {"embed_tokens.weight": torch.tensor(self.reader.tensor("model.embed_tokens.weight")),
                   "lm_head.weight": torch.tensor(self.reader.tensor("lm_head.weight"))}
        save_file(tensors, directory / "teacher.safetensors")
        self.config.save_pretrained(directory)
        write_json(directory / "contract.json", self.contract)
        write_json(directory / 'origin.json', dict(config_identity_path=self.config_identity_path))

    def close(self):
        self.weight_cache.clear()
        self.reader.close()

    def __del__(self):
        if hasattr(self, "reader"):
            self.reader.close()
