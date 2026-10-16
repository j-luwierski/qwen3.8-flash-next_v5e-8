import json
from pathlib import Path

import pytest
import torch

from qwen_tpu.engine import QwenEngine
from qwen_tpu.io import read_json, sha256
from qwen_tpu.shards import capture_worker, make_plan, merge_workers, normalize_record
from qwen_tpu.speculative import greedy_speculate
from qwen_tpu.training import train


@pytest.fixture
def corpus(target, data, tmp_path):
    plan = tmp_path / "plan"
    make_plan(data, plan, 2, tokenizer_identity="synthetic-v1")
    workers = [tmp_path / f"worker-{i}" for i in range(2)]
    for i, directory in enumerate(workers):
        capture_worker(plan, i, target, directory, [0, 2])
    merged = tmp_path / "corpus"
    merge_workers(plan, workers, merged)
    return merged, plan, workers


def test_worker_resume_and_no_overlap(corpus, target):
    merged, plan, workers = corpus
    manifest = read_json(merged / "manifest.json")
    assert len(manifest["items"]) == 4
    assert len({item["key"] for item in manifest["items"]}) == 4
    before = {p.name: sha256(p) for p in (workers[0] / "features").glob("*.ckpt")}
    capture_worker(plan, 0, target, workers[0], [0, 2])
    assert before == {p.name: sha256(p) for p in (workers[0] / "features").glob("*.ckpt")}
    with pytest.raises(ValueError, match="different target"):
        capture_worker(plan, 0, target, workers[0], [1, 2])


def test_merge_rejects_missing_duplicate_and_corruption(corpus, tmp_path):
    _, plan, workers = corpus
    with pytest.raises(ValueError, match="Missing workers"):
        merge_workers(plan, workers[:1], tmp_path / "missing")
    with pytest.raises(ValueError, match="duplicate worker"):
        merge_workers(plan, [workers[0], workers[0]], tmp_path / "duplicate")
    sample = next((workers[1] / "features").glob("*.ckpt"))
    with sample.open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        merge_workers(plan, workers, tmp_path / "corrupted")


def test_merge_recovers_after_interrupted_copy(corpus, tmp_path):
    _, plan, workers = corpus
    staging = tmp_path / "recovered.merging"
    staging.mkdir()
    (staging / "incomplete").write_text("simulated interrupted copy")
    result = merge_workers(plan, workers, tmp_path / "recovered")
    assert len(result["items"]) == 4
    assert not staging.exists()


def test_specforge_reader_consumes_our_features(corpus):
    from specforge.algorithms.common.hidden_states_data import build_offline_reader, normalize_offline_sample
    merged, _, _ = corpus
    refs = build_offline_reader("dflash", str(merged / "features"), run_id="test", ttt_length=4, max_len=256).read()
    assert len(refs) == 4
    sample = torch.load(next((merged / "features").glob("*.ckpt")), weights_only=True)
    normalized = normalize_offline_sample(sample, 256)
    assert normalized["hidden_states"].shape[-1] == 64


def test_real_specforge_resume_and_greedy_equivalence(corpus, target, tmp_path):
    merged, _, _ = corpus
    options = dict(mask_token_id=127, draft_layers=1, block_size=4, anchors=4)
    complete = tmp_path / "complete"
    resumed = tmp_path / "resumed"
    history = train(merged, complete, steps=4, **options)
    train(merged, resumed, steps=2, **options)
    continued = train(merged, resumed, steps=4, resume=resumed / "checkpoint.pt", **options)
    assert history == continued
    a = torch.load(complete / "checkpoint.pt", weights_only=True)
    b = torch.load(resumed / "checkpoint.pt", weights_only=True)
    for key in a["draft"]:
        torch.testing.assert_close(a["draft"][key], b["draft"][key], atol=0, rtol=0)
    assert a["draft"]["candidate_selector.successor_codebook"].abs().sum() > 0
    engine = QwenEngine(target, [0, 2])
    for prompt in ([10, 11, 12], [11], [20, 21, 2, 22]):
        ids, accepted = greedy_speculate(engine, complete / "draft", prompt, 9)
        assert ids == engine.generate(prompt, 9)
        assert all(1 <= count <= 4 for count in accepted)


def test_native_jax_capture_train_speculate(target, data, tmp_path):
    from qwen_tpu.jax_engine import JaxQwenEngine
    plan = tmp_path / "plan"
    make_plan(data, plan, 1, tokenizer_identity="synthetic-v1")
    worker = tmp_path / "worker"
    capture_worker(plan, 0, target, worker, [0, 2], backend="jax")
    corpus = tmp_path / "corpus"
    merge_workers(plan, [worker], corpus)
    out = tmp_path / "train"
    train(corpus, out, steps=1, mask_token_id=127, draft_layers=1, block_size=4, anchors=2)
    engine = JaxQwenEngine(target, [0, 2])
    try:
        actual, _ = greedy_speculate(engine, out / "draft", [10, 11, 12], 4)
        assert actual == engine.generate([10, 11, 12], 4)
    finally:
        engine.close()


@pytest.mark.parametrize("record", [
    {"input_ids": [10, 11], "loss_mask": [0, 1]},
    {"input_ids": [10, -1], "loss_mask": [1, 1]},
    {"input_ids": [10, 11], "loss_mask": [1]},
])
def test_reject_bad_data(record):
    with pytest.raises(ValueError):
        normalize_record(record)
