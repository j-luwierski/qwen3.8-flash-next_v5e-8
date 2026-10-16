import torch
import pytest
from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

from qwen_tpu.engine import QwenEngine, tiny_config


def test_capture_no_leaks_and_causal_prefix(target):
    engine = QwenEngine(target, [0, 1, 3])
    prefix = [10, 11, 12, 2, 14, 15, 16]
    first = engine.capture(prefix, [1] * len(prefix))
    second = engine.capture(prefix + [20, 21, 22], [1] * (len(prefix) + 3))
    assert first["hidden_states"].shape == (1, len(prefix), 96)
    torch.testing.assert_close(first["hidden_states"], second["hidden_states"][:, :len(prefix)], atol=1e-5, rtol=1e-5)
    assert all(len(layer._forward_hooks) == 0 for layer in engine.model.model.layers)


def test_multimodal_checkpoint_text_loader(tmp_path):
    config = Qwen4ExpConfig(text_config=tiny_config().to_dict(), vision_config={
        "depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
        "out_hidden_size": 32, "num_position_embeddings": 16, "patch_size": 2, "spatial_merge_size": 2})
    full = Qwen4ExpForConditionalGeneration(config).eval()
    path = tmp_path / "multimodal"
    full.save_pretrained(path)
    engine = QwenEngine(path, [0, 3])
    ids = torch.tensor([[10, 11, 12]])
    with torch.no_grad():
        torch.testing.assert_close(full(ids).logits, engine.model(ids).logits, atol=1e-6, rtol=1e-5)


def test_bad_layer_and_device(target):
    with pytest.raises(ValueError, match="layer ids"):
        QwenEngine(target, [48])
    with pytest.raises(NotImplementedError, match="CPU reference"):
        QwenEngine(target, [0], device="tpu")
