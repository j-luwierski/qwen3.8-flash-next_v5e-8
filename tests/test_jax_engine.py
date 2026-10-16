import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from qwen_tpu.engine import QwenEngine
from qwen_tpu.jax_engine import JaxQwenEngine


@pytest.mark.parametrize("ids", [[10], [10, 11, 12, 13, 14, 15, 16], [10, 11, 12, 13, 2, 14, 15, 16, 17, 18, 19, 20]])
def test_native_jax_matches_official_model(target, ids):
    reference = QwenEngine(target, [0, 1, 2, 3])
    engine = JaxQwenEngine(target, [0, 1, 2, 3], device="cpu")
    try:
        expected = reference.capture(ids, [1] * len(ids))
        result = engine.capture(ids, [1] * len(ids))
        for key in ("hidden_states", "target_last_hidden_states"):
            torch.testing.assert_close(result[key], expected[key], atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(engine.logits(result["target_last_hidden_states"]),
                                   reference.logits(expected["target_last_hidden_states"]), atol=3e-6, rtol=3e-5)
    finally:
        engine.close()


def test_published_packed_experts_and_multiple_ple_shards(target, tmp_path):
    weights = load_file(target / "model.safetensors")
    for layer in range(4):
        prefix = f"model.layers.{layer}.mlp.experts"
        gu, down = [], []
        for expert in range(4):
            ep = f"{prefix}.{expert}"
            gu.append(torch.cat([weights.pop(ep + ".gate_proj.weight"), weights.pop(ep + ".up_proj.weight")]))
            down.append(weights.pop(ep + ".down_proj.weight"))
        weights[prefix + ".gate_up_proj"] = torch.stack(gu)
        weights[prefix + ".down_proj"] = torch.stack(down)
    p = "model.layers.1.ple.ple_embedding.ngram_embedding."
    table = weights.pop(p + "shard_0.weight")
    for i, chunk in enumerate(table.chunk(4)):
        weights[p + f"shard_{i}.weight"] = chunk.contiguous()
    weights = {k.replace("model.", "model.language_model.", 1) if k.startswith("model.") else k: v for k, v in weights.items()}
    target_config = json.loads((target / "config.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(target_config))
    keys = sorted(weights)
    weight_map = {}
    # Separate physical files exercise the index reader as well as tensor slicing.
    for i, group in enumerate([keys[::2], keys[1::2]]):
        filename = f"part-{i}.safetensors"
        save_file({k: weights[k] for k in group}, tmp_path / filename)
        weight_map.update({k: filename for k in group})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    engine = JaxQwenEngine(tmp_path, [0, 1, 3])
    reference = QwenEngine(target, [0, 1, 3])
    ids = [10, 11, 12, 2, 14, 15, 16]
    try:
        torch.testing.assert_close(engine.capture(ids, [1] * len(ids))["hidden_states"],
                                   reference.capture(ids, [1] * len(ids))["hidden_states"], atol=3e-6, rtol=3e-5)
    finally:
        engine.close()


def test_native_context_guard(target):
    engine = JaxQwenEngine(target, [0], max_tokens=3)
    try:
        with pytest.raises(ValueError, match="1..3"):
            engine.capture([10, 11, 12, 13], [1] * 4)
    finally:
        engine.close()


@pytest.mark.parametrize('query_chunk', [1, 3, 16])
def test_cached_decode_matches_full_forward_across_eos_and_qsa(target, query_chunk):
    engine = JaxQwenEngine(target, [0, 1, 2, 3], query_chunk_size=query_chunk)
    reference = QwenEngine(target, [0, 1, 2, 3])
    ids = [10, 11, 12, 13, 2, 14, 15]
    try:
        state, hidden = engine.prefill(ids)
        for token in [16, 17, 2, 18, 19]:
            state, hidden = engine.decode(token, state)
            ids.append(token)
            expected = reference.capture(ids, [1] * len(ids))['target_last_hidden_states'][:, -1:]
            torch.testing.assert_close(hidden, expected, atol=3e-6, rtol=3e-5)
        options = dict(temperature=.7, top_p=.95, top_k=20, seed=123)
        a = engine.generate([10, 11, 12], 8, **options)
        assert a == engine.generate([10, 11, 12], 8, **options)
        assert a == reference.generate([10, 11, 12], 8, **options)
    finally:
        engine.close()


def test_sparse_attention_supports_more_than_2048_tokens(target, tmp_path):
    import shutil
    larger = tmp_path / 'long-target'
    shutil.copytree(target, larger)
    config = json.loads((larger / 'config.json').read_text())
    config['max_position_embeddings'] = 4096
    (larger / 'config.json').write_text(json.dumps(config))
    engine = JaxQwenEngine(larger, [3], query_chunk_size=16)
    try:
        ids = [10 + i % 100 for i in range(2050)]
        result = engine.capture(ids, [1] * len(ids))
        assert result['hidden_states'].shape == (1, 2050, 32)
        assert torch.isfinite(result['hidden_states']).all()
    finally:
        engine.close()
