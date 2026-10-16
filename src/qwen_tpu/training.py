"""Run SpecForge's actual DFlash2 architecture and objective on CPU/CUDA/JAX.

This small launcher bypasses upstream's GPU-only SGLang/FSDP orchestration.
It does not reimplement the algorithm or average independently trained drafts.
"""

import math
import os
import inspect
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from transformers import Qwen3Config

from .io import atomic_write, digest, read_json, sha256, write_json
from .training_data import TrainingCorpus
from .runtime import SessionPaused, validation_loop


class CPUAdamW:
    """FP32 master parameters and optimizer state on host, BF16 compute on CUDA."""
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.masters = [nn.Parameter(p.detach().float().cpu()) for p in self.parameters]
        self.optimizer = torch.optim.AdamW(self.masters, lr=lr)
        self.param_groups = self.optimizer.param_groups

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            parameter.grad = None
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for master, parameter in zip(self.masters, self.parameters):
            master.grad = parameter.grad.detach().float().cpu() if parameter.grad is not None else None
        self.optimizer.step()
        with torch.no_grad():
            for master, parameter in zip(self.masters, self.parameters):
                parameter.copy_(master.to(parameter.device, dtype=parameter.dtype))

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(), "masters": [p.detach() for p in self.masters]}

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        with torch.no_grad():
            for master, value in zip(self.masters, state["masters"]):
                master.copy_(value)


def draft_config(contract, *, target_layers, mask_token_id, draft_layers=2, block_size=8):
    h = contract["hidden_size"]
    if not 0 <= mask_token_id < contract["vocab_size"]:
        raise ValueError("mask_token_id must be inside target vocabulary")
    if draft_layers < 1 or block_size < 2:
        raise ValueError("draft_layers >= 1 and block_size >= 2 are required")
    head_dim = min(128, h // 4)
    while head_dim >= 2 and (h % head_dim or head_dim % 2):
        head_dim -= 1
    if head_dim < 2:
        raise ValueError("Unsupported hidden dimension")
    heads = h // head_dim
    return Qwen3Config(
        architectures=["DFlash2DraftModel"], hidden_size=h, intermediate_size=h * 4,
        num_hidden_layers=draft_layers, num_attention_heads=heads,
        num_key_value_heads=math.gcd(heads, 4), head_dim=head_dim,
        vocab_size=contract["vocab_size"], num_target_layers=target_layers,
        max_position_embeddings=262144, is_causal=False,
        layer_types=["full_attention"] * draft_layers,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0},
        dflash_config={"block_size": block_size, "target_layer_ids": contract["layer_ids"],
                       "mask_token_id": mask_token_id, "conv_kernel_size": 2,
                       "conv_group_size": math.gcd(h, 16), "selector_rank": min(h, 256),
                       "selector_top_k": min(contract["vocab_size"], 16)},
    )


def load_teacher_assets(corpus):
    corpus = Path(corpus)
    manifest = read_json(corpus / "manifest.json")
    path = corpus / "assets" / "teacher.safetensors"
    if sha256(path) != manifest["assets_sha256"]:
        raise ValueError("Frozen teacher assets were modified")
    if sha256(corpus / "assets" / "config.json") != manifest["assets_config_sha256"]:
        raise ValueError("Teacher config was modified")
    tensors = load_file(path)
    embedding = nn.Embedding.from_pretrained(tensors["embed_tokens.weight"].float(), freeze=True)
    head = nn.Linear(embedding.embedding_dim, embedding.num_embeddings, bias=False)
    head.weight = nn.Parameter(tensors["lm_head.weight"].float(), requires_grad=False)
    return manifest, embedding, head


def train(corpus, out, *, steps=None, mask_token_id, draft_layers=2, block_size=8,
          learning_rate=6e-4, anchors=8, seed=42, resume=None, checkpoint_every=10,
          device="cpu", devices="all", backend="torch", precision="float32", batch_size=1,
          epochs=None, objective_chunk_blocks=32, warmup_steps=0, optimizer_offload=False,
          gradient_checkpointing=False, online_target=None, validation_corpus=None,
          eval_every=100, sequence_length=None, teacher_engine_options=None, control=None):
    if steps is not None and epochs is not None:
        raise ValueError("Choose either steps or epochs")
    if backend == "jax" or device == "tpu":
        from .jax_training import train_jax
        return train_jax(corpus, out, steps=steps, mask_token_id=mask_token_id, draft_layers=draft_layers,
                         block_size=block_size, learning_rate=learning_rate, anchors=anchors, seed=seed,
                         resume=resume, checkpoint_every=checkpoint_every, device=device, devices=devices,
                         precision=precision, batch_size=batch_size, epochs=epochs,
                         objective_chunk_blocks=objective_chunk_blocks, warmup_steps=warmup_steps,
                         online_target=online_target, validation_corpus=validation_corpus,
                         eval_every=eval_every, sequence_length=sequence_length,
                         gradient_checkpointing=gradient_checkpointing, teacher_engine_options=teacher_engine_options, control=control)
    from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
    from specforge.modeling.draft.dflash2 import DFlash2DraftModel

    if (steps is not None and steps < 1) or anchors < 1 or checkpoint_every < 1 or learning_rate <= 0 or batch_size < 1:
        raise ValueError("steps, anchors, checkpoint_every and learning_rate must be positive")
    if (epochs is not None and (type(epochs) is not int or epochs < 1)) or eval_every < 1 or warmup_steps < 0:
        raise ValueError("Invalid epoch count, evaluation interval or warmup")
    if device not in ("cpu", "cuda") or precision not in ("float32", "bfloat16"):
        raise ValueError("Torch training supports cpu/cuda and float32/bfloat16")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a CUDA-enabled PyTorch in a separate environment")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    own_group = world > 1 and not dist.is_initialized()
    if own_group:
        dist.init_process_group("nccl" if device == "cuda" else "gloo")
    source = validation = None
    try:
        torch.manual_seed(seed)
        target_device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if device == "cuda" else torch.device("cpu")
        if device == "cuda":
            torch.cuda.set_device(target_device)
        corpus, out = Path(corpus), Path(out)
        if control and resume is None and (out / 'checkpoint.pt').exists():
            resume = out / 'checkpoint.pt'
        if out.exists() and any(out.iterdir()) and resume is None:
            raise ValueError("Nonempty training directory: use --resume or another output")
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
        if world > 1:
            dist.barrier()
        manifest, embedding, head = load_teacher_assets(corpus)
        source = TrainingCorpus(corpus, online_target=online_target, engine_options=teacher_engine_options)
        if validation_corpus:
            validation = TrainingCorpus(validation_corpus, online_target=online_target, engine_options=teacher_engine_options)
            if validation.contract != source.contract:
                raise ValueError("Validation corpus uses a different target contract")
            source.check_disjoint(validation)
        target_config = read_json(corpus / "assets" / "config.json")
        config = draft_config(manifest["contract"], target_layers=target_config["num_hidden_layers"],
                              mask_token_id=mask_token_id, draft_layers=draft_layers, block_size=block_size)
        draft = DFlash2DraftModel(config)
        if gradient_checkpointing:
            draft.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        wrapped = OnlineDFlashModel(draft, head, embedding, mask_token_id=mask_token_id,
                                   block_size=block_size, attention_backend="eager", num_anchors=anchors,
                                   objective_chunk_blocks=objective_chunk_blocks, teacher_metrics=False)
        wrapped.to(target_device)
        if optimizer_offload:
            wrapped.to(dtype=getattr(torch, precision))
        elif device == "cuda" and precision == "bfloat16":
            head.to(dtype=torch.bfloat16)
            embedding.to(dtype=torch.bfloat16)
        model = DistributedDataParallel(wrapped, device_ids=[target_device.index] if device == "cuda" else None) if world > 1 else wrapped
        optimizer = CPUAdamW(draft.parameters(), learning_rate) if optimizer_offload else torch.optim.AdamW(draft.parameters(), lr=learning_rate)
        contract = {"corpus": digest(manifest), "draft": digest(config.to_dict()), "world_size": world,
                    "seed": seed, "learning_rate": learning_rate, "anchors": anchors,
                    "objective": "SpecForge.OnlineDFlashModel.dflash+selector.v1",
                    "specforge_draft_code": sha256(inspect.getfile(DFlash2DraftModel)),
                    "specforge_objective_code": sha256(inspect.getfile(OnlineDFlashModel))}
        contract.update(precision=precision, batch_size=batch_size, warmup_steps=warmup_steps,
                        optimizer_offload=optimizer_offload, objective_chunk_blocks=objective_chunk_blocks,
                        sequence_length=sequence_length, gradient_checkpointing=gradient_checkpointing)
        start, history = 0, []
        if resume:
            saved = torch.load(resume, map_location="cpu", weights_only=True)
            if saved["contract"] != contract:
                raise ValueError("Resume contract differs (corpus/config/world size/optimizer/seed)")
            expected_validation = digest(validation.manifest) if validation else None
            if saved.get('validation_identity', expected_validation) != expected_validation:
                raise ValueError('Validation corpus changed during resume')
            draft.load_state_dict(saved["draft"])
            optimizer.load_state_dict(saved["optimizer"])
            start, history = saved["step"], saved["history"]
            torch.set_rng_state(saved["rng_states"][rank])
            if device == "cuda":
                torch.cuda.set_rng_state(saved["cuda_rng_states"][rank], target_device)
        else:
            # DDP construction synchronizes parameters. Distinct anchor sampling
            # streams per rank, reproducibly saved in each checkpoint.
            torch.manual_seed(seed + rank)
        per_epoch = math.ceil(len(source.items) / (world * batch_size))
        steps = steps if steps is not None else per_epoch * (epochs or 1)
        if steps < start:
            raise ValueError("--steps cannot be smaller than the resumed step")
        items = manifest["items"]
        if len(items) < world:
            raise ValueError("There must be at least one sample per training rank")
        pending_validation = saved.get('validation_progress') if resume else None
        completed_step = start
        context = lambda: torch.autocast(target_device.type, dtype=torch.bfloat16) if precision == 'bfloat16' else nullcontext()

        def stop_requested():
            stop = bool(control and control.should_stop())
            if world > 1 and control:
                flag = torch.tensor(int(stop), device=target_device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                stop = bool(flag.item())
            return stop

        def save_checkpoint():
            states = [None] * world
            rng = torch.get_rng_state()
            if world > 1:
                dist.all_gather_object(states, rng)
            else:
                states[0] = rng
            checkpoint = dict(step=completed_step, draft=draft.state_dict(), optimizer=optimizer.state_dict(),
                              rng_states=states, contract=contract, history=history,
                              validation_progress=pending_validation,
                              validation_identity=digest(validation.manifest) if validation else None)
            if device == 'cuda':
                cuda_states = [None] * world
                if world > 1:
                    dist.all_gather_object(cuda_states, torch.cuda.get_rng_state(target_device))
                else:
                    cuda_states[0] = torch.cuda.get_rng_state(target_device)
                checkpoint['cuda_rng_states'] = cuda_states
            if rank == 0:
                atomic_write(out / 'checkpoint.pt', lambda f: torch.save(checkpoint, f))
                write_json(out / 'metrics.json', history)
                write_json(out / 'progress.json', dict(step=completed_step, total_steps=steps,
                           stage='validation' if pending_validation else 'training',
                           validation_index=pending_validation['next_index'] if pending_validation else None))
            if control:
                control.saved()

        def pause_if_due():
            if stop_requested():
                save_checkpoint()
                raise SessionPaused('validation' if pending_validation else 'training', control.reason)

        def finish_validation():
            nonlocal pending_validation
            if pending_validation is None:
                return
            def evaluate(index):
                pause_if_due()
                if control and control.due():
                    save_checkpoint()
                # Each validation example has its own seed. The training RNG
                # is restored even if a session ends in the middle of validation.
                devices_rng = [target_device.index] if device == 'cuda' else []
                with torch.random.fork_rng(devices=devices_rng), torch.no_grad(), context():
                    torch.manual_seed(seed + index)
                    sample = {k: v.to(target_device) for k, v in validation.batch([index]).items()}
                    return wrapped(**sample, collect_detailed_metrics=False)[0].item()
            try:
                result = validation_loop(pending_validation, len(validation.items), evaluate,
                                         None, save_checkpoint)
            except SessionPaused:
                raise
            history[-1]['validation_loss'] = result
            pending_validation = None

        # A session may end after the final optimizer step, during validation.
        finish_validation()
        cached_epoch, order = None, None
        draft.train()
        for step in range(start, steps):
            pause_if_due()
            epoch, offset = divmod(step, per_epoch)
            if epoch != cached_epoch:
                order = torch.randperm(len(items), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
                cached_epoch = epoch
            indices = [order[(offset * world * batch_size + rank * batch_size + j) % len(items)] for j in range(batch_size)]
            batch = source.batch(indices, padded_length=sequence_length)
            width = min(anchors, int(((batch["loss_mask"][:, :-1] > .5) & (batch["loss_mask"][:, 1:] > .5)).sum(1).max()))
            batch = {k: value.to(target_device) for k, value in batch.items()}
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * min(1., (step + 1) / max(1, warmup_steps))
            optimizer.zero_grad(set_to_none=True)
            context = lambda: torch.autocast(target_device.type, dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext()
            with context():
                loss, accuracy, _ = model(**batch, max_valid_anchors=width, collect_detailed_metrics=False)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            metrics = torch.tensor([loss.detach().item(), accuracy.detach().item(), grad_norm.item()], device=target_device)
            if world > 1:
                dist.all_reduce(metrics)
                metrics /= world
            log = {"step": step + 1, "loss": metrics[0].item(), "accuracy": metrics[1].item(),
                   "grad_norm": metrics[2].item()}
            completed_step = step + 1
            history.append(log)
            if validation and ((step + 1) % eval_every == 0 or step + 1 == steps):
                pending_validation = dict(step=completed_step, next_index=0, losses=[])
                finish_validation()
            if rank == 0:
                print(log, flush=True)
            if (step + 1) % checkpoint_every == 0 or step + 1 == steps or (control and control.due()) or stop_requested():
                save_checkpoint()
            if step + 1 < steps:
                pause_if_due()
        pause_if_due()
        if rank == 0:
            draft.save_pretrained(out / "draft")
            write_json(out / "draft" / "target_contract.json", manifest["contract"])
            write_json(out / "metrics.json", history)
            write_json(out / "training_contract.json", contract)
        if world > 1:
            dist.barrier()
        return history
    finally:
        if source:
            source.close()
        if validation:
            validation.close()
        if own_group:
            dist.destroy_process_group()
