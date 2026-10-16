"""Pure device kernels: no readers, Python callbacks or host activations."""

from functools import lru_cache, partial


@lru_cache(maxsize=None)
def hyper_connection_kernel(count, hidden_size, eps, inject):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def kernel(streams, norm_weight, down, up, injection_weight):
        grouped = streams.astype(jnp.float32).reshape(*streams.shape[:-1], count, hidden_size)
        grouped *= jax.lax.rsqrt(jnp.mean(grouped ** 2, -1, keepdims=True) + eps)
        normed = (grouped.reshape(streams.shape) * (norm_weight.astype(jnp.float32) + 1)).astype(streams.dtype)
        weights = jax.nn.sigmoid(jax.nn.silu((normed @ down.T) / count) @ up.T)
        mixed = (weights * normed).reshape(*streams.shape[:-1], count, hidden_size).mean(-2)
        if inject:
            return mixed, 2 * jax.nn.sigmoid((normed @ injection_weight.T) / count)
        return mixed
    return kernel


@lru_cache(maxsize=None)
def convolution_kernel(dilation):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def kernel(padded, weights):
        length = padded.shape[0] - dilation * (weights.shape[-1] - 1)
        out = jnp.zeros((length, padded.shape[-1]), padded.dtype)
        for tap in range(weights.shape[-1]):
            out += padded[tap * dilation:tap * dilation + length] * weights[:, tap]
        return jax.nn.silu(out)
    return kernel


@lru_cache(maxsize=None)
def sparse_qsa_kernel(ratio, budget, query_tile):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def attention(q, iq, k, v, pooled, positions):
        # Query arrays have already been padded to a multiple of query_tile.
        heads, dim = q.shape[1:]
        kv_heads = k.shape[1]
        groups = heads // kv_heads
        blocks = pooled.shape[0]
        take = min(blocks, budget // ratio)

        def tile(args):
            query, index_query, pos = args
            complete = (pos + 1) // ratio
            if take:
                scores = jax.nn.relu(jnp.einsum('qhd,bd->qbh', index_query.astype(jnp.float32),
                                                pooled.astype(jnp.float32))).sum(-1) / iq.shape[-1] ** .5
                scores = jnp.where(jnp.arange(blocks)[None] < complete[:, None], scores, -jnp.inf)
                _, selected_blocks = jax.lax.top_k(scores, take)
                selected = (selected_blocks[..., None] * ratio + jnp.arange(ratio)).reshape(query_tile, -1)
                valid = jnp.repeat(selected_blocks < complete[:, None], ratio, axis=-1)
            else:
                selected = jnp.empty((query_tile, 0), jnp.int32)
                valid = jnp.empty((query_tile, 0), bool)
            tail = complete[:, None] * ratio + jnp.arange(ratio - 1)[None]
            selected = jnp.concatenate((selected, tail), axis=-1)
            valid = jnp.concatenate((valid, tail <= pos[:, None]), axis=-1)
            # Invalid padded slots must never index outside the KV allocation.
            selected = jnp.clip(selected, 0, k.shape[0] - 1)
            keys, values = k[selected], v[selected]
            query = query.reshape(query_tile, kv_heads, groups, dim)
            scores = jnp.einsum('qhgd,qkhd->qhgk', query, keys).astype(jnp.float32) / dim ** .5
            probs = jax.nn.softmax(jnp.where(valid[:, None, None], scores, -jnp.inf), axis=-1).astype(q.dtype)
            return jnp.einsum('qhgk,qkhd->qhgd', probs, values).reshape(query_tile, heads, dim)

        tiles = q.shape[0] // query_tile
        result = jax.lax.map(tile, (q.reshape(tiles, query_tile, *q.shape[1:]),
                                    iq.reshape(tiles, query_tile, *iq.shape[1:]),
                                    positions.reshape(tiles, query_tile)))
        return result.reshape(q.shape)
    return attention


@lru_cache(maxsize=None)
def device_kernels():
    import jax
    import jax.numpy as jnp

    @jax.jit
    def linear(x, weight):
        return x @ weight.T

    @jax.jit
    def rms_norm(x, weight, eps, offset):
        dtype = x.dtype
        value = x.astype(jnp.float32)
        value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + eps)
        return (value * (weight.astype(jnp.float32) + offset)).astype(dtype)

    @jax.jit
    def expert(x, gate_up, down, weights):
        gate, up = jnp.split(x @ gate_up.T, 2, axis=-1)
        return ((jax.nn.silu(gate) * up) @ down.T) * weights[:, None]

    @partial(jax.jit, donate_argnums=(0, 1, 2))
    def update_kv(old_k, old_v, old_raw, k, v, raw, offset):
        return (jax.lax.dynamic_update_slice(old_k, k, (offset, 0, 0)),
                jax.lax.dynamic_update_slice(old_v, v, (offset, 0, 0)),
                jax.lax.dynamic_update_slice(old_raw, raw, (offset, 0)))

    return linear, rms_norm, expert, update_kv
