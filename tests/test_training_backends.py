import os
import subprocess
import sys
import textwrap

import pytest
import torch

from qwen_tpu.io import read_json
from qwen_tpu.shards import capture_worker, make_plan, merge_workers
from qwen_tpu.training import train


@pytest.fixture
def training_corpus(target, data, tmp_path):
    plan, worker, corpus = [tmp_path / name for name in ('plan', 'worker', 'corpus')]
    make_plan(data, plan, 1, tokenizer_identity='test')
    capture_worker(plan, 0, target, worker, [0, 2])
    merge_workers(plan, [worker], corpus)
    return corpus


def test_torch_bfloat16_offload_batching_and_checkpointing(training_corpus, tmp_path):
    out = tmp_path / 'offload'
    history = train(training_corpus, out, steps=2, mask_token_id=127,
                    draft_layers=1, block_size=4, anchors=2, precision='bfloat16',
                    optimizer_offload=True, batch_size=2, gradient_checkpointing=True)
    assert len(history) == 2 and all(torch.isfinite(torch.tensor(row['loss'])) for row in history)
    train(training_corpus, out, steps=2, mask_token_id=127, draft_layers=1,
          block_size=4, anchors=2, precision='bfloat16', optimizer_offload=True,
          batch_size=2, gradient_checkpointing=True, resume=out / 'checkpoint.pt')
    with pytest.raises(ValueError, match='overlapping'):
        train(training_corpus, tmp_path / 'leaky', steps=1, mask_token_id=127,
              validation_corpus=training_corpus)


def test_real_specforge_jax_loss_and_gradients_match_torch(training_corpus):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torchax
    from specforge.modeling.draft.dflash2 import DFlash2DraftModel
    from qwen_tpu.jax_training import SpecForgeLoss
    from qwen_tpu.training import draft_config, load_teacher_assets
    from qwen_tpu.training_data import TrainingCorpus
    manifest, embedding, head = load_teacher_assets(training_corpus)
    config = draft_config(manifest['contract'], target_layers=4, mask_token_id=127,
                          draft_layers=1, block_size=4)
    torch.manual_seed(9)
    model = SpecForgeLoss(DFlash2DraftModel(config), head, embedding, anchors=1,
                         block_size=4, mask_token_id=127, objective_chunk_blocks=1)
    source = TrainingCorpus(training_corpus)
    batch = source.batch([0])
    # Exactly one valid anchor eliminates differences in random generators.
    batch['loss_mask'].zero_()
    batch['loss_mask'][:, -2:] = 1
    loss, _ = model(batch['input_ids'], batch['hidden_states'], batch['loss_mask'], 1)
    loss.backward()
    expected = {k: p.grad.detach().numpy().copy() for k, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    env = torchax.default_env()
    states, function = torchax.extract_jax(model, env=env)
    params = {k: v for k, v in states.items() if k in expected}
    fixed = {k: v for k, v in states.items() if k not in params}
    args = tuple(jnp.asarray(batch[k].numpy()) for k in ('input_ids', 'hidden_states', 'loss_mask')) + (1,)
    def objective(weights):
        with env.override_property(prng=jax.random.key(9)):
            return function({**fixed, **weights}, args)[0]
    actual, grads = jax.jit(jax.value_and_grad(objective))(params)
    np.testing.assert_allclose(float(actual), loss.item(), atol=2e-6, rtol=2e-5)
    for k in expected:
        np.testing.assert_allclose(np.asarray(grads[k]), expected[k], atol=3e-6, rtol=3e-4, err_msg=k)


def test_jax_eight_devices_training_placement_and_exact_resume(training_corpus, tmp_path):
    script = textwrap.dedent('''
        import sys, torch
        from pathlib import Path
        from qwen_tpu.training import train
        from qwen_tpu.io import read_json
        torch.set_num_threads(1)
        corpus, root = map(Path, sys.argv[1:])
        options = dict(mask_token_id=127, draft_layers=1, block_size=4, anchors=2,
                       backend='jax', device='cpu', devices='all', objective_chunk_blocks=2,
                       gradient_checkpointing=True, precision='bfloat16')
        full, resumed = root / 'full', root / 'resumed'
        expected = train(corpus, full, steps=3, **options)
        train(corpus, resumed, steps=1, **options)
        actual = train(corpus, resumed, steps=3, resume=resumed / 'checkpoint.pt', **options)
        assert actual == expected
        a = torch.load(full / 'checkpoint.pt', weights_only=True)
        b = torch.load(resumed / 'checkpoint.pt', weights_only=True)
        for key in a['params']:
            torch.testing.assert_close(a['params'][key], b['params'][key], atol=0, rtol=0)
        layout = read_json(full / 'devices.json')
        assert len(layout['devices']) == 8
        import math
        assert any(len(entry['local_shapes']) == 8 and
                   math.prod(entry['local_shapes'][0]) * 8 == math.prod(entry['shape'])
                   for entry in layout['parameter_sharding'].values())
        print('eight device sharded training and exact resume passed')
    ''')
    env = dict(os.environ, JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1',
               XLA_FLAGS='--xla_force_host_platform_device_count=8 --xla_cpu_multi_thread_eigen=false')
    result = subprocess.run([sys.executable, '-c', script, str(training_corpus), str(tmp_path)],
                            env=env, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'eight device sharded training' in result.stdout
