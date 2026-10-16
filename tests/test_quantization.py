import numpy as np
import pytest
import torch

from qwen_tpu.io import checkpoint_identity
from qwen_tpu.jax_engine import JaxQwenEngine, TensorReader
from qwen_tpu.quantization import prepare_weights, quantize, dequantize


@pytest.mark.parametrize('bits', [4, 8])
def test_packed_groups_slicing_and_quantized_teacher(target, tmp_path, bits):
    x = torch.randn(3, 7, 71)
    packed = quantize(x, bits, 64)
    actual = dequantize(packed['values'], packed['scales'], bits, 64, 71)
    maximum = (1 << (bits - 1)) - 1
    assert ((x - actual).abs() <= x.abs().amax(-1, keepdim=True) / maximum + 1e-6).all()
    out = tmp_path / 'quant'
    report = prepare_weights(target, out, bits=bits, chunk_mib=1)
    assert report['output_bytes'] < report['source_bytes']
    identity = checkpoint_identity(out)
    assert identity != checkpoint_identity(target)
    prepare_weights(target, out, bits=bits, chunk_mib=1)
    assert identity == checkpoint_identity(out)
    source, quant = TensorReader(target), TensorReader(out)
    try:
        for key in ('model.embed_tokens.weight', 'lm_head.weight', 'model.norm.weight'):
            if key in source.mapping:
                np.testing.assert_array_equal(quant.tensor(key), source.tensor(key))
        key = 'model.layers.0.mlp.experts.0.gate_proj.weight'
        value = quant.tensor(key)
        np.testing.assert_array_equal(quant.rows(key, [2, 1, 2]), value[[2, 1, 2]])
        assert len(quant.files) <= 32
    finally:
        source.close()
        quant.close()
    engine, original = JaxQwenEngine(out, [0, 2]), JaxQwenEngine(target, [0, 2])
    try:
        ids = [10, 11, 12, 13, 14, 15]
        hidden = engine.capture(ids, [1] * len(ids))['target_last_hidden_states']
        expected = original.capture(ids, [1] * len(ids))['target_last_hidden_states']
        assert torch.isfinite(hidden).all()
        # This verifies quantization error, not exact equivalence of teachers.
        assert (hidden - expected).square().mean().sqrt() < (.08 if bits == 4 else .01)
        assert len(engine.generate(ids, 2)) >= len(ids) + 1
    finally:
        engine.close()
        original.close()
