"""Train the actual SpecForge DFlash2 module/objective through Torchax + JAX.

Parameters and Adam moments are sharded across local devices; frozen embedding
and LM-head matrices are sharded too. The same code runs on logical CPU devices
for verification and on TPU without a separate hand-written DFlash2 objective.
"""

import inspect
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .io import atomic_write, digest, sha256, write_json
from .training_data import TrainingCorpus
from .runtime import SessionPaused, validation_loop


class SpecForgeLoss(nn.Module):
    def __init__(self, draft, head, embedding, *, anchors, block_size, mask_token_id, objective_chunk_blocks):
        super().__init__()
        from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
        self.wrapped = OnlineDFlashModel(
            draft, head, embedding, mask_token_id=mask_token_id, block_size=block_size,
            num_anchors=anchors, attention_backend="eager", teacher_metrics=False,
            objective_chunk_blocks=objective_chunk_blocks)

    def forward(self, input_ids, hidden_states, loss_mask, width):
        loss, accuracy, _ = self.wrapped(input_ids=input_ids, hidden_states=hidden_states,
                                         loss_mask=loss_mask, max_valid_anchors=width,
                                         collect_detailed_metrics=False)
        return loss, accuracy


def train_jax(corpus, out, *, steps, mask_token_id, draft_layers=2, block_size=8,
              learning_rate=6e-4, anchors=8, seed=42, resume=None, checkpoint_every=10,
              device="cpu", devices="all", precision="float32", batch_size=1,
              epochs=None, objective_chunk_blocks=32, warmup_steps=0,
              online_target=None, validation_corpus=None, eval_every=100,
              sequence_length=None, gradient_checkpointing=False, teacher_engine_options=None, control=None, **_unused):
    import jax
    import jax.numpy as jnp
    import torchax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from specforge.modeling.draft.dflash2 import DFlash2DraftModel
    from .devices import selected_device_count
    from .training import draft_config, load_teacher_assets

    if precision not in ("float32", "bfloat16") or batch_size < 1 or anchors < 1:
        raise ValueError("JAX training requires float32/bfloat16, positive batch_size and anchors")
    if epochs is not None and (type(epochs) is not int or epochs < 1):
        raise ValueError("epochs must be a positive integer")
    if eval_every < 1 or warmup_steps < 0 or learning_rate <= 0:
        raise ValueError("Invalid evaluation interval, warmup or learning rate")
    count_devices = selected_device_count("jax", device, devices)
    mesh = Mesh(np.asarray(jax.local_devices(backend=device)[:count_devices]), ("model",))
    replicated = NamedSharding(mesh, P())
    source = TrainingCorpus(corpus, online_target=online_target, target_device=device, target_devices=devices,
                            engine_options=teacher_engine_options)
    validation = None
    try:
        manifest, embedding, head = load_teacher_assets(corpus)
        out = Path(out)
        if control and not resume and (out / 'checkpoint.pt').exists():
            resume = out / 'checkpoint.pt'
        if out.exists() and any(out.iterdir()) and not resume:
            raise ValueError("Nonempty training directory: use --resume or another output")
        out.mkdir(parents=True, exist_ok=True)
        per_epoch = math.ceil(len(source.items) / batch_size)
        total_steps = steps if steps is not None else per_epoch * (epochs or 1)
        if total_steps < 1 or checkpoint_every < 1:
            raise ValueError("Positive step count and checkpoint interval required")
        from .io import read_json
        target_config = read_json(Path(corpus) / "assets/config.json")
        config = draft_config(manifest["contract"], target_layers=target_config["num_hidden_layers"],
                              mask_token_id=mask_token_id, draft_layers=draft_layers, block_size=block_size)
        torch.manual_seed(seed)
        draft = DFlash2DraftModel(config)
        model = SpecForgeLoss(draft, head, embedding, anchors=anchors, block_size=block_size,
                              mask_token_id=mask_token_id, objective_chunk_blocks=objective_chunk_blocks)
        env = torchax.default_env()
        env.config.use_int32_for_index = True
        with jax.default_device(mesh.devices.flat[0]):
            states, function = torchax.extract_jax(model, env=env)
        train_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

        def shard(value):
            partition = P("model", *([None] * (value.ndim - 1))) if (
                value.ndim >= 2 and value.shape[0] % count_devices == 0) else P()
            return jax.device_put(value, NamedSharding(mesh, partition))

        compute_dtype = getattr(jnp, precision)
        params = {k: shard(v.astype(jnp.float32)) for k, v in states.items() if k in train_names}
        frozen = {k: shard(v.astype(compute_dtype) if jnp.issubdtype(v.dtype, jnp.floating) else v)
                  for k, v in states.items() if k not in train_names}
        del states
        first = jax.tree.map(jnp.zeros_like, params)
        second = jax.tree.map(jnp.zeros_like, params)
        contract = {"backend": "jax-specforge-v1", "corpus": digest(manifest), "config": digest(config.to_dict()),
                    "precision": precision, "seed": seed, "anchors": anchors, "batch_size": batch_size,
                    "learning_rate": learning_rate, "warmup_steps": warmup_steps,
                    "objective_chunk_blocks": objective_chunk_blocks,
                    "sequence_length": sequence_length, "gradient_checkpointing": gradient_checkpointing,
                    "specforge_code": sha256(inspect.getfile(type(model.wrapped))),
                    "specforge_draft_code": sha256(inspect.getfile(DFlash2DraftModel)),
                    "torchax_version": torchax.__version__}
        history, start = [], 0
        if resume:
            saved = torch.load(resume, map_location="cpu", weights_only=True)
            if saved["contract"] != contract:
                raise ValueError("JAX resume contract differs")
            for field, values in (("params", params), ("first", first), ("second", second)):
                for key in values:
                    values[key] = shard(saved[field][key].numpy())
            start, history = saved["step"], saved["history"]
        if total_steps < start:
            raise ValueError("Total steps cannot be smaller than the resumed step")
        padded = sequence_length or max(item["length"] for item in source.items)
        if validation_corpus:
            validation = TrainingCorpus(validation_corpus, online_target=online_target, target_device=device, target_devices=devices,
                                        engine_options=teacher_engine_options)
            if validation.contract != source.contract:
                raise ValueError("Validation corpus uses a different target/capture contract")
            source.check_disjoint(validation)
        if resume:
            expected_validation = digest(validation.manifest) if validation else None
            if saved.get('validation_identity', expected_validation) != expected_validation:
                raise ValueError('Validation corpus changed during resume')

        def loss_function(weights, fixed, key, batch, width):
            computed = {k: value.astype(compute_dtype) for k, value in weights.items()}
            with env.override_property(prng=key):
                return function({**fixed, **computed}, (batch["input_ids"], batch["hidden_states"], batch["loss_mask"], width))

        def update(weights, m, v, fixed, key, batch, rate, step, width):
            objective = jax.checkpoint(loss_function, static_argnums=(4,)) if gradient_checkpointing else loss_function
            (loss, accuracy), grads = jax.value_and_grad(objective, has_aux=True)(weights, fixed, key, batch, width)
            norm = jnp.sqrt(sum(jnp.sum(g.astype(jnp.float32) ** 2) for g in jax.tree.leaves(grads)))
            grads = jax.tree.map(lambda g: g * jnp.minimum(1., 1. / (norm + 1e-6)), grads)
            m = jax.tree.map(lambda old, g: .9 * old + .1 * g, m, grads)
            v = jax.tree.map(lambda old, g: .999 * old + .001 * g ** 2, v, grads)
            weights = jax.tree.map(lambda w, a, b: w * (1 - rate * .01) - rate * (a / (1 - .9 ** step)) /
                                   (jnp.sqrt(b / (1 - .999 ** step)) + 1e-8), weights, m, v)
            return weights, m, v, jnp.stack([loss, accuracy, norm])

        state_layout = jax.tree.map(lambda value: value.sharding, params)
        # Reuse the old buffers for updated parameters/moments. Otherwise an
        # Adam step can temporarily require a second full copy of all states.
        step_fn = jax.jit(update, static_argnums=(8,), donate_argnums=(0, 1, 2),
                          out_shardings=(state_layout, state_layout, state_layout, replicated))
        eval_fn = jax.jit(loss_function, static_argnums=(4,))
        pending_validation = saved.get('validation_progress') if resume else None
        completed_step = start

        def save_checkpoint():
            host = lambda tree: {k: torch.from_numpy(np.asarray(v).copy()) for k, v in tree.items()}
            saved_state = dict(step=completed_step, params=host(params), first=host(first), second=host(second),
                               contract=contract, history=history, validation_progress=pending_validation,
                               validation_identity=digest(validation.manifest) if validation else None)
            atomic_write(out / 'checkpoint.pt', lambda f: torch.save(saved_state, f))
            write_json(out / 'metrics.json', history)
            write_json(out / 'progress.json', dict(step=completed_step, total_steps=total_steps,
                stage='validation' if pending_validation else 'training',
                validation_index=pending_validation['next_index'] if pending_validation else None))
            if control:
                control.saved()

        def pause_if_due():
            if control and control.should_stop():
                save_checkpoint()
                raise SessionPaused('validation' if pending_validation else 'training', control.reason)

        def finish_validation():
            nonlocal pending_validation
            if pending_validation is None:
                return
            val_length = max(padded, max(item['length'] for item in validation.items))
            def evaluate(index):
                arrays = validation.batch_jax([index], padded_length=val_length, sharding=replicated)
                return float(eval_fn(params, frozen, jax.device_put(jax.random.key(seed+index), replicated),
                                     arrays, min(anchors, val_length-1))[0])
            try:
                result = validation_loop(pending_validation, len(validation.items), evaluate, control, save_checkpoint)
            except SessionPaused:
                save_checkpoint()
                raise
            history[-1]['validation_loss'] = result
            pending_validation = None

        finish_validation()
        cached_epoch, order = None, None
        for step in range(start, total_steps):
            pause_if_due()
            epoch, offset = divmod(step, per_epoch)
            if epoch != cached_epoch:
                order = torch.randperm(len(source.items), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
                cached_epoch = epoch
            indices = order[offset * batch_size:(offset + 1) * batch_size]
            # Fixed batch shapes keep the compiled TPU program reusable.
            indices += [indices[-1]] * (batch_size - len(indices))
            width = min(anchors, padded - 1)
            arrays = source.batch_jax(indices, padded_length=padded, sharding=replicated)
            rate = learning_rate * min(1., (step + 1) / max(1, warmup_steps))
            params, first, second, values = step_fn(params, first, second, frozen,
                jax.device_put(jax.random.key(seed + step), replicated), arrays,
                jax.device_put(np.float32(rate), replicated), jax.device_put(np.int32(step + 1), replicated), width)
            values = np.asarray(values)
            if not np.isfinite(values).all():
                raise FloatingPointError("Non-finite JAX training statistics")
            log = {"step": step + 1, "epoch": epoch + 1, "loss": float(values[0]),
                   "accuracy": float(values[1]), "grad_norm": float(values[2])}
            completed_step = step + 1
            history.append(log)
            if validation and ((step + 1) % eval_every == 0 or step + 1 == total_steps):
                pending_validation = dict(step=completed_step, next_index=0, losses=[])
                finish_validation()
            print(log, flush=True)
            if (step + 1) % checkpoint_every == 0 or step + 1 == total_steps or (control and (control.due() or control.should_stop())):
                save_checkpoint()
            if step + 1 < total_steps:
                pause_if_due()
        pause_if_due()
        export = draft.state_dict()
        for key, value in params.items():
            name = key.removeprefix("wrapped.draft_model.")
            if name in export:
                export[name] = torch.from_numpy(np.asarray(value).copy())
        draft.load_state_dict(export)
        draft.save_pretrained(out / "draft")
        write_json(out / "draft/target_contract.json", manifest["contract"])
        write_json(out / "training_contract.json", contract)
        write_json(out / "metrics.json", history)
        write_json(out / "devices.json", {"backend": device, "devices": [str(d) for d in mesh.devices.flat],
                                         "parameter_sharding": {k: {"shape": list(value.shape), "spec": str(value.sharding),
                                                                    "local_shapes": [list(s.data.shape) for s in value.addressable_shards]}
                                                                for k, value in params.items()}})
        return history
    finally:
        source.close()
        if validation:
            validation.close()
