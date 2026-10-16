"""Single-sequence streaming MoE with device transfers and collective reduction.

Only routing IDs (small integer metadata) visit the host to select disk weights.
Activations, routing probabilities and partial results remain JAX device arrays.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .devices import CaptureReplicas


class ExpertParallelEngine:
    def __init__(self, model_path, layers, *, devices="all", **options):
        import jax
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        self.replicas = CaptureReplicas(model_path, layers, backend="jax", devices=devices, **options)
        self.coordinator = self.replicas.engines[0]
        self.pool = ThreadPoolExecutor(max_workers=self.replicas.count, thread_name_prefix="qwen-experts")
        self.mesh = Mesh(np.asarray([engine.device for engine in self.replicas.engines]), ('expert',))
        self.replicated = NamedSharding(self.mesh, P())
        self.partial_sharding = NamedSharding(self.mesh, P('expert', None, None))

        def reduce_partials(partials):
            return jax.lax.psum(partials[0], 'expert')
        self.reduce_partials = jax.jit(jax.shard_map(reduce_partials, mesh=self.mesh,
            in_specs=P('expert', None, None), out_specs=P(), check_vma=False))
        if self.replicas.count > 1:
            self.coordinator._expert_dispatch = self._dispatch
        self.expert_calls_per_device = [0] * self.replicas.count

    def __getattr__(self, name):
        return getattr(self.coordinator, name)

    @staticmethod
    def _partial(engine, x, prefix, ids, weights, experts):
        return engine._routed_experts(x, prefix, ids, weights, experts)

    def _dispatch(self, x, prefix, ids, weights):
        jax = self.coordinator.jax
        experts = np.unique(ids)
        count = min(self.replicas.count, len(experts))
        # Replicate directly between devices, without np.asarray/device_get.
        xs = jax.device_put(x, self.replicated)
        ws = jax.device_put(weights, self.replicated)
        x_local = {s.device: s.data for s in xs.addressable_shards}
        w_local = {s.device: s.data for s in ws.addressable_shards}
        futures = []
        for index in range(count):
            engine = self.replicas.engines[index]
            self.expert_calls_per_device[index] += 1
            futures.append(self.pool.submit(self._partial, engine, x_local[engine.device],
                                           prefix, ids, w_local[engine.device], experts[index::count]))
        try:
            partials = [future.result() for future in futures]
            for engine in self.replicas.engines[count:]:
                with jax.default_device(engine.device):
                    partials.append(self.coordinator.jnp.zeros(x.shape, x.dtype))
            array = jax.make_array_from_single_device_arrays(
                (self.replicas.count, *x.shape), self.partial_sharding,
                [value[None] for value in partials])
            reduced = self.reduce_partials(array)
            return next(s.data for s in reduced.addressable_shards if s.device == self.coordinator.device)
        except BaseException:
            for future in futures:
                try:
                    future.result()
                except BaseException:
                    pass
            raise

    def close(self):
        self.pool.shutdown(wait=True)
        self.coordinator._expert_dispatch = None
        self.replicas.stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
