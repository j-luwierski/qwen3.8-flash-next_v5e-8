"""Single-process, per-device capture replicas; no multi-process TPU runtime.

Each replica owns its reader and JAX arrays. Only completed CPU feature tensors
leave its thread. This does not split a sequence or a traced function across
threads, and leaves the cross-account plan/identity independent of device count.
"""

from contextlib import ExitStack

from .engine import load_engine
from .io import checkpoint_identity


def selected_device_count(backend, platform, selection):
    if selection != "all":
        try:
            count = int(selection)
        except (ValueError, TypeError):
            raise ValueError("--devices must be 'all' or a positive device count") from None
        if str(count) != str(selection) or count < 1:
            raise ValueError("--devices must be 'all' or a positive device count")
    else:
        count = None
    if backend == "jax":
        import jax
        available = len(jax.local_devices(backend=platform))
    elif backend == "torch":
        if platform != "cpu":
            raise ValueError("The torch reference backend requires CPU")
        available = 1
    else:
        raise ValueError(f"Unknown backend: {backend}")
    if available == 0 or (count is not None and count > available):
        raise ValueError(f"Requested {selection} devices, but {available} local {platform} devices are available")
    return available if count is None else count


class CaptureReplicas:
    def __init__(self, model_path, layers, *, backend, device, devices="all", **options):
        self.count = selected_device_count(backend, device, devices)
        self.stack = ExitStack()
        self.engines = []
        try:
            # A large checkpoint is hashed once per account worker, not once per
            # core. This value never comes from the input manifest or user cache.
            weight_id = checkpoint_identity(model_path) if backend == "jax" else None
            for index in range(self.count):
                placement = {"device_index": index, "_weight_identity": weight_id} if backend == "jax" else {}
                engine = load_engine(model_path, layers, backend=backend, device=device, devices="1", **placement, **options)
                self.stack.callback(engine.close)
                self.engines.append(engine)
            self.contract = self.engines[0].contract
            if any(engine.contract != self.contract for engine in self.engines):
                raise ValueError("Local capture replicas have incompatible contracts")
        except BaseException:
            self.stack.close()
            raise

    @property
    def device_names(self):
        return [str(getattr(engine, "device", "cpu")) for engine in self.engines]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.stack.close()
