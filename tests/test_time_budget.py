import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from qwen_tpu.generation import generate
from qwen_tpu.io import read_json, sha256
from qwen_tpu.runtime import RunControl, SessionPaused, resumable_copy
from qwen_tpu.shards import capture_worker, make_plan, merge_workers
from qwen_tpu.training import train
from qwen_tpu.workflow import run_job


class Clock:
    def __init__(self):
        self.value = 0
    def __call__(self):
        return self.value
    def advance(self, seconds=1):
        self.value += seconds


class SmallEngine:
    config = SimpleNamespace(max_position_embeddings=128, eos_token_id=None)
    def __init__(self, clock=None):
        self.clock = clock
        self.prefills = []
    def prefill(self, ids):
        self.prefills.append(list(ids))
        return {}, torch.zeros(1, 1, 8)
    def decode(self, token, state):
        if self.clock:
            self.clock.advance()
        return state, torch.zeros(1, 1, 8)
    def logits(self, hidden):
        return torch.arange(8, dtype=torch.float32)[None] / 4


def test_generation_resumes_mid_response_with_exact_rng():
    clock, progress = Clock(), []
    options = dict(temperature=.9, top_p=.95, top_k=6, seed=19)
    expected = generate(SmallEngine(), [1, 2], 9, **options)
    engine = SmallEngine(clock)
    with pytest.raises(SessionPaused):
        generate(engine, [1, 2], 9, control=RunControl(3, 0, 1, clock=clock),
                 on_progress=lambda p: progress.append(json.loads(json.dumps(p))), **options)
    saved = progress[-1]
    assert 2 < len(saved['input_ids']) < len(expected)
    resumed = SmallEngine()
    actual = generate(resumed, [1, 2], 9, progress=saved, **options)
    assert actual == expected and resumed.prefills == [saved['input_ids']]
    with pytest.raises(ValueError, match='request differs'):
        generate(SmallEngine(), [1, 2], 9, progress=saved, temperature=.1)


def test_budget_reserves_save_margin_and_restores_signal_handlers():
    import signal
    clock = Clock()
    control = RunControl(10, 2, 3, clock=clock)
    original = signal.getsignal(signal.SIGTERM)
    with control:
        clock.advance(7)
        assert not control.should_stop()
        clock.advance(1)
        assert control.should_stop()
    assert signal.getsignal(signal.SIGTERM) == original
    with pytest.raises(ValueError):
        RunControl(10, 0, 0)


def config_for(target, tmp_path):
    prompts = tmp_path / 'prompts.jsonl'
    prompts.write_text(''.join(json.dumps({'prompt': f't{10+i} t20'}) + '\n' for i in range(4)))
    return dict(model=str(target), out=str(tmp_path / 'job'), workers=1,
                backend='torch', device='cpu', dtype='float32', layers=[0, 2],
                data=dict(data=str(prompts), samples=4, seed=42, validation_fraction=0),
                generation=dict(max_new_tokens=6, temperature=.7, stop_token_ids=[]),
                max_length=32, feature_mode='tokens', pack_worker=True)


def test_worker_pause_snapshot_restore_new_path_and_continue_tokens(target, tmp_path, monkeypatch):
    from qwen_tpu.engine import QwenEngine
    clock = Clock()
    original = QwenEngine.decode
    def timed_decode(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        clock.advance()
        return result
    monkeypatch.setattr(QwenEngine, 'decode', timed_decode)
    config = config_for(target, tmp_path)
    root = run_job(config, stage='worker', _control=RunControl(2, 0, 1, clock=clock))
    status = read_json(root / 'session.json')
    assert status['status'] == 'paused'
    archive = Path(status['resume_archive'])
    assert archive.exists() and not (root / 'worker-0/train/manifest.json').exists()
    saved = read_json(next((root / 'worker-0/train/generation').glob('*.json')))['progress']
    assert not saved['complete'] and len(saved['input_ids']) > 2
    moved = dict(config)
    moved['out'] = str(tmp_path / 'restored')
    # Model/data mounts can change names without changing the immutable inputs.
    moved_prompts = tmp_path / 'same-data-new-mount.jsonl'
    moved_prompts.write_bytes(Path(config['data']['data']).read_bytes())
    moved['data'] = {**config['data'], 'data': str(moved_prompts)}
    import shutil
    moved_model = tmp_path / 'same-model-new-mount'
    shutil.copytree(target, moved_model)
    moved['model'] = str(moved_model)
    moved['runtime'] = dict(resume_from=str(archive))
    run_job(moved, stage='worker')
    restored = Path(moved['out'])
    assert read_json(restored / 'session.json')['status'] == 'complete'
    baseline = {**config, 'out': str(tmp_path / 'baseline')}
    run_job(baseline, stage='worker')
    full = Path(baseline['out'])
    for path in (full / 'worker-0/train/features').glob('*.ckpt'):
        a = torch.load(path, weights_only=True)
        b = torch.load(restored / 'worker-0/train/features' / path.name, weights_only=True)
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    # Fresh training can consume worker artifacts even when the target mount
    # differs from the generation session's mount.
    from qwen_tpu.training_data import TrainingCorpus
    portable_corpus = tmp_path / 'portable-corpus'
    merge_workers(full / 'plan', [full / 'worker-0/train'], portable_corpus)
    source = TrainingCorpus(portable_corpus, online_target=moved_model)
    try:
        assert source.sample(0)['hidden_states'].shape[-1] == 64
    finally:
        source.close()
    # Reusing resume_from must not overwrite newer files in the writable job.
    marker = restored / 'newer-file'
    marker.write_text('preserve')
    run_job(moved, stage='worker')
    assert marker.read_text() == 'preserve'


@pytest.fixture
def corpus(target, data, tmp_path):
    plan, worker, out = [tmp_path / n for n in ('plan', 'worker', 'corpus')]
    make_plan(data, plan, 1, tokenizer_identity='test')
    capture_worker(plan, 0, target, worker, [0, 2])
    merge_workers(plan, [worker], out)
    return out, plan, worker


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_timed_training_exact_resume(corpus, tmp_path, monkeypatch, backend):
    from qwen_tpu.training_data import TrainingCorpus
    source, _, _ = corpus
    clock = Clock()
    if backend == 'torch':
        original = torch.optim.AdamW.step
        def timed(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            clock.advance(10)
            return result
        monkeypatch.setattr(torch.optim.AdamW, 'step', timed)
    else:
        original = TrainingCorpus.batch_jax
        def timed(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            clock.advance(10)
            return result
        monkeypatch.setattr(TrainingCorpus, 'batch_jax', timed)
    options = dict(steps=3, mask_token_id=127, draft_layers=1, block_size=4,
                   anchors=2, backend=backend, objective_chunk_blocks=2, checkpoint_every=99)
    paused, complete = tmp_path / 'paused', tmp_path / 'complete'
    with pytest.raises(SessionPaused):
        train(source, paused, control=RunControl(5, 0, 1, clock=clock), **options)
    assert torch.load(paused / 'checkpoint.pt', weights_only=True)['step'] == 1
    actual = train(source, paused, control=RunControl(None, clock=clock), **options)
    expected = train(source, complete, **options)
    assert actual == expected
    for key, value in load_file(complete / 'draft/model.safetensors').items():
        torch.testing.assert_close(value, load_file(paused / 'draft/model.safetensors')[key], atol=0, rtol=0)


@pytest.mark.parametrize('backend', ['torch', 'jax'])
def test_validation_cursor_and_rng_resume(corpus, target, data, tmp_path, monkeypatch, backend):
    from qwen_tpu.training_data import TrainingCorpus
    source, _, _ = corpus
    records = [json.loads(line) for line in data.read_text().splitlines()]
    for row in records:
        row['id'] = 'val-' + row['id']
    val_data = tmp_path / 'val.jsonl'
    val_data.write_text(''.join(json.dumps(r)+'\n' for r in records))
    make_plan(val_data, tmp_path / 'val-plan', 1, tokenizer_identity='test')
    capture_worker(tmp_path / 'val-plan', 0, target, tmp_path / 'val-worker', [0, 2])
    val = tmp_path / 'val'
    merge_workers(tmp_path / 'val-plan', [tmp_path / 'val-worker'], val)
    clock, seen = Clock(), []
    name = 'batch_jax' if backend == 'jax' else 'batch'
    original = getattr(TrainingCorpus, name)
    def timed(self, indices, **kwargs):
        result = original(self, indices, **kwargs)
        if self.path == val:
            seen.extend(indices)
            clock.advance()
        return result
    monkeypatch.setattr(TrainingCorpus, name, timed)
    options = dict(steps=2, mask_token_id=127, draft_layers=1, block_size=4, anchors=2,
                   backend=backend, objective_chunk_blocks=2, validation_corpus=val, eval_every=1)
    out = tmp_path / 'val-paused'
    with pytest.raises(SessionPaused):
        train(source, out, control=RunControl(2, 0, 1, clock=clock), **options)
    saved = torch.load(out / 'checkpoint.pt', weights_only=True)
    assert saved['step'] == 1 and saved['validation_progress']['next_index'] == 2
    seen.clear()
    actual = train(source, out, control=RunControl(None, clock=clock), **options)
    assert seen[:2] == [2, 3]
    expected = train(source, tmp_path / 'val-full', **options)
    assert actual == expected


def test_merge_continues_verified_files(corpus, tmp_path, monkeypatch):
    import qwen_tpu.shards as shards
    _, plan, worker = corpus
    original = shards.resumable_copy
    clock = Clock()
    def timed(*args, **kwargs):
        original(*args, **kwargs)
        clock.advance(10)
    monkeypatch.setattr(shards, 'resumable_copy', timed)
    out = tmp_path / 'merge-paused'
    with pytest.raises(SessionPaused):
        merge_workers(plan, [worker], out, control=RunControl(5, 0, 1, clock=clock))
    staging = out.with_name(out.name+'.merging')
    first = next((staging / 'features').glob('*.ckpt'))
    inode = first.stat().st_ino
    merge_workers(plan, [worker], out)
    assert (out / 'features' / first.name).stat().st_ino == inode


def test_copy_resumes_at_byte_offset_and_verifies_checksum(tmp_path):
    class Stops(RunControl):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def should_stop(self):
            self.calls += 1
            return self.calls >= 3
    src, dest = tmp_path / 'source', tmp_path / 'copy'
    src.write_bytes(bytes(range(64)))
    with pytest.raises(SessionPaused):
        resumable_copy(src, dest, sha256(src), Stops(), chunk_bytes=8)
    assert dest.with_name('copy.copying').stat().st_size == 16
    resumable_copy(src, dest, sha256(src), chunk_bytes=8)
    assert src.read_bytes() == dest.read_bytes()


def test_local_prompt_fetch_continues_at_saved_offset(tmp_path, monkeypatch):
    import qwen_tpu.data as data
    source, out = tmp_path / 'source.jsonl', tmp_path / 'prompts.jsonl'
    source.write_text(''.join(json.dumps({'prompt':str(i)})+'\n' for i in range(8)))
    clock = Clock()
    original = data.normalize_prompt
    def timed(raw):
        result = original(raw)
        clock.advance()
        return result
    monkeypatch.setattr(data, 'normalize_prompt', timed)
    with pytest.raises(SessionPaused):
        data.fetch_prompts(out, data=source, samples=8, control=RunControl(3, 0, 1, clock=clock))
    saved = read_json(out.with_suffix('.fetch.json'))
    assert len(saved['selected']) == 3 and saved['position'] > 0
    actual = data.fetch_prompts(out, data=source, samples=8)
    assert len(actual) == len({row['id'] for row in actual}) == 8


def test_partial_zip_extraction_survives_snapshot_and_restore(tmp_path):
    import zipfile
    from qwen_tpu.session_io import unpack_cached, snapshot_job, restore_job
    source = tmp_path / 'worker.zip'
    blob = bytes(range(256)) * (64*1024)  # 16 MiB, split at the 8 MiB copy boundary.
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('large.bin', blob)
    job = tmp_path / 'job'
    extracted = job / 'artifact-cache/cache'
    class StopAfterChunk(RunControl):
        def should_stop(self):
            part = extracted / 'large.bin.extracting'
            return part.exists() and part.stat().st_size >= 8*1024**2
    with pytest.raises(SessionPaused):
        unpack_cached(source, extracted, StopAfterChunk())
    assert (extracted / 'large.bin.extracting').stat().st_size == 8*1024**2
    snapshot = snapshot_job(job, tmp_path / 'resume.zip')
    restored = restore_job(snapshot, tmp_path / 'restored')
    result = unpack_cached(source, restored / 'artifact-cache/cache')
    assert (result / 'large.bin').read_bytes() == blob


def test_quantization_pause_reuses_completed_chunks(target, tmp_path, monkeypatch):
    import qwen_tpu.quantization as quant
    clock = Clock()
    original = quant.quantize
    def timed(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.advance(5)
        return result
    monkeypatch.setattr(quant, 'quantize', timed)
    out = tmp_path / 'quant'
    with pytest.raises(SessionPaused):
        quant.prepare_weights(target, out, control=RunControl(2, 0, 1, clock=clock))
    first = next((out / 'weights').glob('*.safetensors'))
    inode = first.stat().st_ino
    quant.prepare_weights(target, out)
    assert first.stat().st_ino == inode
    assert (out / 'qwen_quantization.json').exists()


def test_cli_budget_flags_pause_and_restart_same_command(target, tmp_path):
    import os, subprocess, sys
    config = config_for(target, tmp_path)
    config['runtime'] = dict(archive_on_pause=False)
    path = tmp_path / 'job.json'
    path.write_text(json.dumps(config))
    base = [sys.executable, '-m', 'qwen_tpu.cli', 'run', '--config', str(path), '--stage', 'worker']
    env = dict(os.environ, JAX_PLATFORMS='cpu')
    stopped = subprocess.run(base+['--max-run-seconds', '0', '--save-margin-seconds', '0'],
                             capture_output=True, text=True, env=env, timeout=60)
    assert stopped.returncode == 0, stopped.stdout+stopped.stderr
    assert read_json(Path(config['out']) / 'session.json')['status'] == 'paused'
    completed = subprocess.run(base, capture_output=True, text=True, env=env, timeout=90)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    assert read_json(Path(config['out']) / 'session.json')['status'] == 'complete'


def test_archive_failure_preserves_resume_directory(target, tmp_path, monkeypatch):
    import qwen_tpu.workflow as workflow
    config = config_for(target, tmp_path)
    def failed(*args, **kwargs):
        raise OSError('simulated disk full')
    monkeypatch.setattr(workflow, 'snapshot_job', failed)
    root = run_job(config, stage='worker', _control=RunControl(0, 0))
    status = read_json(root / 'session.json')
    assert status['status'] == 'paused' and 'disk full' in status['archive_error']
    assert (root / 'job-config.json').exists()
