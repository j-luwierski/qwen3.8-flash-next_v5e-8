import numpy as np
import pytest
import torch

from qwen_tpu.jax_engine import JaxQwenEngine
from qwen_tpu.jax_kernels import sparse_qsa_kernel


@pytest.mark.parametrize('length,ratio,budget,tile', [(1, 2, 4, 4), (17, 2, 4, 4), (19, 4, 8, 8), (9, 1, 3, 4)])
def test_compiled_sparse_qsa_matches_canonical_oracle_without_host_transfer(length, ratio, budget, tile):
    import jax
    import jax.numpy as jnp
    rng = np.random.default_rng(11)
    q = rng.normal(size=(length, 4, 8)).astype('float32')
    iq = rng.normal(size=(length, 2, 8)).astype('float32')
    k, v = [rng.normal(size=(length, 2, 8)).astype('float32') for _ in range(2)]
    pooled = rng.normal(size=(length // ratio, 8)).astype('float32')
    # Include exact zero-score ties at the selection boundary.
    pooled[::2] = 0
    expected = []
    for pos in range(length):
        complete = (pos + 1) // ratio
        scores = np.maximum(np.einsum('hd,bd->bh', iq[pos], pooled[:complete]), 0).sum(-1) / 8 ** .5
        chosen = np.lexsort((np.arange(complete), -scores))[:budget // ratio]
        selected = np.concatenate(((chosen[:, None] * ratio + np.arange(ratio)).reshape(-1),
                                   np.arange(complete * ratio, pos + 1)))
        keys = np.repeat(k[selected], 2, axis=1)
        values = np.repeat(v[selected], 2, axis=1)
        logits = np.einsum('hd,khd->hk', q[pos], keys) / 8 ** .5
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        expected.append(np.einsum('hk,khd->hd', probs, values))
    padded = (length + tile - 1) // tile * tile
    pad = lambda x: np.pad(x, ((0, padded-length), *[(0, 0)]*(x.ndim-1)))
    args = [jax.device_put(x) for x in (pad(q), pad(iq), k, v, pooled, pad(np.arange(length, dtype='int32')))]
    kernel = sparse_qsa_kernel(ratio, budget, tile)
    with jax.transfer_guard_device_to_host('disallow'):
        actual = kernel(*args)
        actual.block_until_ready()
    np.testing.assert_allclose(np.asarray(actual)[:length], expected, atol=4e-6, rtol=3e-5)
    hlo = kernel.lower(*args).as_text().lower()
    assert 'python_callback' not in hlo and 'xla_python' not in hlo


def test_optimized_cache_decode_and_weight_reuse(target):
    engine = JaxQwenEngine(target, [0, 2], qsa_backend='jax', weight_cache_mib=16,
                           query_chunk_size=4, kv_cache_block_size=8)
    try:
        ids = [10, 11, 12, 2, 13]
        state, hidden = engine.prefill(ids)
        prefix = 'model.layers.3.self_attn'
        assert state[prefix]['arrays'][0].shape[0] == 8
        for token in (14, 15, 2, 16, 17):
            state, hidden = engine.decode(token, state)
            ids.append(token)
            expected = engine.capture(ids, [1]*len(ids))['target_last_hidden_states'][:, -1:]
            np.testing.assert_allclose(np.asarray(hidden), expected.numpy(), atol=4e-6, rtol=4e-5)
        assert state[prefix]['arrays'][0].shape[0] == 16
        before = engine.weight_cache.stats()
        engine.capture(ids, [1]*len(ids))
        after = engine.weight_cache.stats()
        assert after['hits'] > before['hits']
        assert after['host_to_device_weight_bytes'] == before['host_to_device_weight_bytes']
        assert 0 < after['resident_bytes'] <= 16 * 1024**2
        assert engine.contract['qsa_selection'] == 'jax.top_k.lower_index.v1'
        assert engine.generate([10, 11, 12], 4) == engine.generate([10, 11, 12], 4)
    finally:
        engine.close()
    assert engine.weight_cache.stats()['resident_bytes'] == 0


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_optimized_matches_reference_when_all_keys_are_selected(target, dtype):
    engine = JaxQwenEngine(target, [0, 1, 2, 3], qsa_backend='jax', query_chunk_size=4, dtype=dtype)
    reference = JaxQwenEngine(target, [0, 1, 2, 3], qsa_backend='reference', dtype=dtype)
    try:
        ids = [10, 11, 12, 13]
        a, b = engine.capture(ids, [1]*4), reference.capture(ids, [1]*4)
        torch.testing.assert_close(a['hidden_states'], b['hidden_states'],
                                   atol=.02 if dtype == 'bfloat16' else 4e-6,
                                   rtol=.02 if dtype == 'bfloat16' else 4e-5)
        assert engine.contract['backend'] != reference.contract['backend']
    finally:
        engine.close()
        reference.close()
