import json
import shutil
import zipfile
from contextlib import ExitStack

import pytest
import torch
from safetensors.torch import load_file

from qwen_tpu.data import ChatTokenizer, fetch_prompts
from qwen_tpu.io import digest, read_json, sha256, write_json
from qwen_tpu.shards import capture_worker, read_plan
from qwen_tpu.workflow import artifact_directory, prepare_job, run_job


def job_config(target, tmp_path):
    source = tmp_path / 'prompts.jsonl'
    source.write_text(''.join(json.dumps({'prompt': f't{i} t{10+i} t20'}) + '\n' for i in range(10, 16)))
    return {'model': str(target), 'out': str(tmp_path / 'job'), 'workers': 2,
            'data': {'data': str(source), 'samples': 6, 'validation_fraction': .2},
            'backend': 'jax', 'device': 'cpu', 'devices': '1', 'dtype': 'float32',
            'engine_options': {'qsa_backend': 'jax', 'weight_cache_mib': 4, 'query_chunk_size': 4},
            'layers': [0, 2], 'chat_template_kwargs': {'enable_thinking': True},
            'generation': {'max_new_tokens': 4, 'temperature': .7, 'top_p': .9},
            'max_length': 32, 'feature_mode': 'tokens', 'pack_worker': True,
            'training': {'backend': 'jax', 'steps': 2, 'mask_token_id': 127,
                         'draft_layers': 1, 'block_size': 4, 'anchors': 2,
                         'objective_chunk_blocks': 2, 'eval_every': 1}}


def test_full_prompt_job_archives_online_teacher_and_real_jax_training(target, tmp_path, monkeypatch):
    config = job_config(target, tmp_path)
    root = prepare_job(config)
    ids = {item['key'] for item in read_plan(root / 'plan')['items']}
    assert not ids & {item['key'] for item in read_plan(root / 'validation-plan')['items']}
    for worker in range(2):
        run_job(config, stage='worker', worker=worker)
    transcript = next((root / 'worker-0/train/conversations').glob('*.json'))
    generated = read_json(transcript)['generated']
    tokenizer = ChatTokenizer(target, template_kwargs=config['chat_template_kwargs'])
    assert generated['input_ids'][:generated['prompt_tokens']] == tokenizer.prefix(generated)
    assert '<think>' in generated['prompt_raw']
    assert generated['loss_mask'] == [0] * generated['prompt_tokens'] + [1] * generated['response_tokens']
    before = sha256(transcript)
    resumed = run_job(config, stage='worker', worker=0)
    assert read_json(resumed / 'train/manifest.json')['execution']['resumed_samples'] == 3
    assert sha256(transcript) == before
    artifacts = [root / f'worker-{i}.zip' for i in range(2)]
    shutil.rmtree(root / 'worker-0')
    shutil.rmtree(root / 'worker-1')
    run_job(config, stage='merge', artifacts=artifacts)
    assert read_json(root / 'corpus/manifest.json')['feature_mode'] == 'tokens'
    for path in (root / 'corpus/features').glob('*.ckpt'):
        assert set(torch.load(path, weights_only=True)) == {'input_ids', 'loss_mask'}
    # Online training must not materialize floating teacher activations on CPU.
    import jax
    import numpy as np
    from jax.sharding import SingleDeviceSharding
    from qwen_tpu.training_data import TrainingCorpus
    source = TrainingCorpus(root / 'corpus', online_target=target, engine_options=config['engine_options'])
    array_type = type(jax.device_put(np.array([1.], dtype='float32')))
    original_array = array_type.__array__
    def guarded_array(array, *args, **kwargs):
        if np.issubdtype(array.dtype, np.floating):
            raise AssertionError('Floating teacher activation copied to host')
        return original_array(array, *args, **kwargs)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(array_type, '__array__', guarded_array)
            batch = source.batch_jax([0], padded_length=32, sharding=SingleDeviceSharding(jax.local_devices()[0]))
            jax.block_until_ready(batch)
        assert batch['hidden_states'].shape == (1, 32, 64)
    finally:
        source.close()
    draft = run_job(config, stage='train')
    assert (draft / 'model.safetensors').exists()
    metrics = read_json(root / 'training/metrics.json')
    assert len(metrics) == 2 and all('validation_loss' in row for row in metrics)
    before = load_file(draft / 'model.safetensors')
    # Completed checkpoint can be re-exported, even after losing export files.
    shutil.rmtree(draft)
    run_job(config, stage='train')
    for key, tensor in load_file(draft / 'model.safetensors').items():
        torch.testing.assert_close(tensor, before[key], atol=0, rtol=0)


def test_interrupted_preparation_and_conversation_integrity(target, tmp_path):
    config = job_config(target, tmp_path)
    root = prepare_job(config)
    original = read_plan(root / 'plan')
    (root / 'preparation.json').unlink()
    assert prepare_job(config) == root
    assert read_plan(root / 'plan') == original
    run_job(config, stage='worker', worker=0)
    transcript = next((root / 'worker-0/train/conversations').glob('*.json'))
    envelope = read_json(transcript)
    envelope['generated']['input_ids'][-1] = 42
    write_json(transcript, envelope)
    with pytest.raises(ValueError, match='conversation'):
        run_job(config, stage='worker', worker=0)
    # Pending transcript is protected too, before a feature receipt exists.
    receipt = root / 'worker-0/train/receipts' / transcript.name
    receipt.unlink()
    with pytest.raises(ValueError, match='conversation'):
        run_job(config, stage='worker', worker=0)


def test_prompt_filtering_skips_context_and_preserves_provenance(target, tmp_path):
    config = job_config(target, tmp_path)
    config['workers'] = 1
    config['max_length'] = 10
    config['data']['validation_fraction'] = 0
    source = tmp_path / 'prompts.jsonl'
    source.write_text(json.dumps({'prompt': 't10 ' * 40}) + '\n' + json.dumps({'prompt': 't11'}) + '\n')
    root = run_job(config, stage='worker')
    manifest = read_json(root / 'train/manifest.json')
    assert sum(row.get('status') == 'skipped' for row in manifest['items']) == 1
    run_job(config, stage='merge')
    merged = read_json(root.parent / 'corpus/manifest.json')
    assert len(merged['items']) == len(merged['skipped']) == 1


def test_archive_rejects_path_traversal(tmp_path):
    path = tmp_path / 'bad.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('../escaped', b'bad')
    with ExitStack() as stack, pytest.raises(ValueError, match='Unsafe'):
        artifact_directory(path, stack)
    assert not (tmp_path / 'escaped').exists()
