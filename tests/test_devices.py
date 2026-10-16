import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from qwen_tpu.devices import selected_device_count


@pytest.mark.parametrize("selection", ["0", "-1", "foo", "1.5", True])
def test_invalid_device_selection(selection):
    with pytest.raises(ValueError, match="positive device count"):
        selected_device_count("jax", "cpu", selection)


def test_reference_backend_rejects_multiple_devices():
    with pytest.raises(ValueError, match="1 local cpu"):
        selected_device_count("torch", "cpu", "2")


def test_eight_cpu_devices_capture_resume_and_failure(target, tmp_path):
    # A fresh interpreter is necessary: the CPU device count is fixed at JAX
    # initialization. These are real JAX CPU devices, not mocked TPU devices.
    script = textwrap.dedent('''
        import json, sys, threading
        from pathlib import Path
        import jax, torch
        from qwen_tpu.jax_engine import JaxQwenEngine
        from qwen_tpu.shards import make_plan, capture_worker, merge_workers
        from qwen_tpu.io import sha256

        torch.set_num_threads(1)
        root, target = Path(sys.argv[1]), Path(sys.argv[2])
        assert len(jax.local_devices(backend="cpu")) == 8
        records = [{"id": str(i), "input_ids": [10+i, 11, 12, 13, 14, 15, 16],
                    "loss_mask": [0, 0, 1, 1, 1, 1, 1]} for i in range(8)]
        data = root / "data.jsonl"
        data.write_text("".join(json.dumps(record) + "\\n" for record in records))
        plan = root / "plan"
        make_plan(data, plan, 1, tokenizer_identity="synthetic-v1")
        original = JaxQwenEngine.capture
        original_put = JaxQwenEngine._put
        seen, threads, lock = set(), set(), threading.Lock()
        barrier = threading.Barrier(8, timeout=120)

        def placed(self, value):
            array = original_put(self, value)
            assert array.device == self.device
            return array

        def synchronized(self, *args, **kwargs):
            with lock:
                seen.add(self.device.id)
                threads.add(threading.get_ident())
            barrier.wait()
            return original(self, *args, **kwargs)

        JaxQwenEngine._put = placed
        JaxQwenEngine.capture = synchronized
        parallel = root / "parallel"
        result = capture_worker(plan, 0, target, parallel, [0, 2], backend="jax", devices="all")
        assert len(seen) == len(threads) == 8
        assert result["execution"]["samples_per_device"] == [1] * 8
        assert len(result["items"]) == 8
        JaxQwenEngine.capture = original
        serial = root / "serial"
        capture_worker(plan, 0, target, serial, [0, 2], backend="jax", devices="1")
        for path in (parallel / "features").glob("*.ckpt"):
            a = torch.load(path, weights_only=True)
            b = torch.load(serial / "features" / path.name, weights_only=True)
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=3e-6, rtol=3e-5)
        before = {p.name: sha256(p) for p in (parallel / "features").glob("*.ckpt")}
        resumed = capture_worker(plan, 0, target, parallel, [0, 2], backend="jax", devices="1")
        assert resumed["execution"]["resumed_samples"] == 8
        assert resumed["execution"]["samples_per_device"] == [0]
        assert before == {p.name: sha256(p) for p in (parallel / "features").glob("*.ckpt")}
        merge_workers(plan, [parallel], root / "merged")

        def fail_one(self, *args, **kwargs):
            if self.device.id == 3:
                raise RuntimeError("injected device failure")
            return original(self, *args, **kwargs)

        JaxQwenEngine.capture = fail_one
        failed = root / "failed"
        try:
            capture_worker(plan, 0, target, failed, [0, 2], backend="jax", devices="all")
        except RuntimeError as exc:
            assert "injected device failure" in str(exc)
        else:
            raise AssertionError("Device failure was swallowed")
        assert not (failed / "manifest.json").exists()
        assert len(list((failed / "receipts").glob("*.json"))) == 7
        JaxQwenEngine.capture = original
        recovered = capture_worker(plan, 0, target, failed, [0, 2], backend="jax", devices="all")
        assert recovered["execution"]["resumed_samples"] == 7
        assert sum(recovered["execution"]["samples_per_device"]) == 1

        # A single sequence also uses eight devices for routed MoE experts.
        # Ten active experts per token matches the full target's top-k width.
        from transformers import Qwen4ExpForCausalLM
        from qwen_tpu.engine import tiny_config, load_engine
        config = tiny_config()
        config.num_experts, config.num_experts_per_tok = 16, 10
        wide = root / "wide-target"
        torch.manual_seed(42)
        Qwen4ExpForCausalLM(config).save_pretrained(wide)
        reference = JaxQwenEngine(wide, [0, 2])
        with load_engine(wide, [0, 2], backend="jax", device="cpu", devices="all") as parallel_engine:
            ids = [10, 11, 12, 13, 14, 15, 16]
            a = parallel_engine.capture(ids, [1] * len(ids))
            b = reference.capture(ids, [1] * len(ids))
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=3e-6, rtol=3e-5)
            assert all(count > 0 for count in parallel_engine.expert_calls_per_device)
            assert parallel_engine.generate([10, 11, 12], 3) == reference.generate([10, 11, 12], 3)
        reference.close()
        # Optimized route: guard both the coordinator and worker threads against
        # device->host transfers of activations, probabilities and partial sums.
        with load_engine(wide, [0, 2], backend="jax", device="cpu", devices="all",
                         qsa_backend="jax", weight_cache_mib=4, query_chunk_size=4) as ep:
            import numpy as np
            original_partial = ep._partial
            def guarded_partial(*args):
                with jax.transfer_guard_device_to_host("disallow"):
                    return original_partial(*args)
            ep._partial = guarded_partial
            x = jax.device_put(np.random.default_rng(42).normal(size=(3, 32)).astype("float32"), ep.coordinator.device)
            ids = np.tile(np.arange(10), (3, 1))
            weights = jax.device_put(np.full((3, 10), .1, dtype="float32"), ep.coordinator.device)
            expected = ep.coordinator._routed_experts(x, "model.layers.0.mlp", ids, weights, np.arange(10))
            with jax.transfer_guard_device_to_host("disallow"):
                actual = ep._dispatch(x, "model.layers.0.mlp", ids, weights)
                actual.block_until_ready()
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=3e-6, rtol=3e-5)
            array = jax.device_put(np.zeros((8, 3, 32), dtype="float32"), ep.partial_sharding)
            hlo = ep.reduce_partials.lower(array).as_text().lower()
            assert "all_reduce" in hlo and "python_callback" not in hlo
            assert all(count > 0 for count in ep.expert_calls_per_device)
        print("eight-device CPU capture, placement, parity, resume and failure recovery passed")
    ''')
    env = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1",
               XLA_FLAGS="--xla_force_host_platform_device_count=8 --xla_cpu_multi_thread_eigen=false")
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(target)],
                            env=env, text=True, capture_output=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "eight-device CPU capture" in result.stdout
