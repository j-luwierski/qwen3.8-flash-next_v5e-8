"""Qwen4Exp reference execution, deliberately independent from GLM's kernels.

Flash-Next has HC residual streams, QSA, DeltaNet and PLE. Reusing GLM's
weight loader or attention equations would silently compute a different model.
The CPU implementation here uses the official Transformers implementation.
"""

from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, Qwen4ExpForCausalLM, Qwen4ExpTextConfig

from .io import checkpoint_identity, digest, write_json

FEATURE_POLICY = "qwen4_exp.post_decoder.final_hc_mixer.v1"


def config_identity(config, origin=None):
    values = config.to_dict()
    if origin is not None and values.get('_name_or_path'):
        values['_name_or_path'] = str(origin)
    return digest(values)


def tiny_config():
    """Real model classes, reduced dimensions; not pretrained weights."""
    return Qwen4ExpTextConfig(
        vocab_size=128, hidden_size=32, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "indexed_attention"],
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        hc_count=4, hc_lowrank=8, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=16, shared_expert_intermediate_size=16,
        ple_layer_ids=[2], ple_embed_dim=32, ple_conv_kernel_size=2,
        ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=31,
        make_ngram_vocab_size_divisible_by=8, split_ngram_parts=1,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=4, indexer_compress_ratio=2,
        max_position_embeddings=256, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
        output_gate_type="sigmoid",
    )


def make_tiny(path, seed=42):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Refusing to overwrite model directory: {path}")
    torch.manual_seed(seed)
    model = Qwen4ExpForCausalLM(tiny_config())
    model.save_pretrained(path)
    # A real local tokenizer/chat template makes the entire data pipeline
    # testable without downloading any pretrained model or tokenizer.
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
             "<think>": 4, "</think>": 5, "<user>": 6, "<assistant>": 7, "<system>": 8}
    vocab.update({f"t{i}": i for i in range(9, 128)})
    tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>",
                                        eos_token="<eos>", pad_token="<pad>")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<' + message['role'] + '> ' + message['content'] + ' ' }}"
        "{% if message['role'] == 'assistant' %}{{ '<eos> ' }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant> ' }}"
        "{% if enable_thinking|default(false) %}{{ '<think> ' }}{% endif %}{% endif %}")
    tokenizer.save_pretrained(path)
    write_json(path / "TEST_MODEL.json", {"random_weights": True, "seed": seed,
                                         "purpose": "CPU correctness smoke tests only"})
    return path


class QwenEngine:
    def __init__(self, model_path, layer_ids, *, revision=None, dtype="float32", device="cpu", config_identity_path=None):
        if device != "cpu":
            raise NotImplementedError(
                "Flash-Next currently has a CPU reference backend only. The GLM JAX/TPU "
                "kernels cannot execute Qwen4Exp; a native quantized TPU port is still required.")
        if dtype not in ("float32", "bfloat16"):
            raise ValueError("Supported CPU dtypes: float32, bfloat16")
        config = AutoConfig.from_pretrained(model_path, revision=revision, trust_remote_code=False)
        text_config = config.get_text_config()
        if text_config.model_type != "qwen4_exp_text":
            raise ValueError(f"Expected Qwen3.8-Flash-Next (qwen4_exp_text), got {text_config.model_type}")
        self.layer_ids = list(layer_ids)
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids) or any(
                i < 0 or i >= text_config.num_hidden_layers for i in self.layer_ids):
            raise ValueError("Capture layer ids must be unique zero-based decoder indices")
        is_local = Path(model_path).is_dir()
        if not is_local and (revision is None or len(revision) != 40):
            raise ValueError("For remote weights pass an immutable 40-character HF commit via --revision")
        if is_local:
            weight_id = checkpoint_identity(model_path)
        else:
            weight_id = revision
        # Drop the vision tower: this adapter intentionally supports text only.
        if config.model_type == "qwen4_exp":
            model, loading = Qwen4ExpForCausalLM.from_pretrained(
                model_path, revision=revision, config=text_config,
                dtype=getattr(torch, dtype), attn_implementation="eager", trust_remote_code=False,
                output_loading_info=True)
        else:
            model, loading = AutoModelForCausalLM.from_pretrained(
                model_path, revision=revision, dtype=getattr(torch, dtype),
                attn_implementation="eager", trust_remote_code=False, output_loading_info=True)
        if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise ValueError(f"Incomplete target weights: {loading}")
        self.model = model.eval().requires_grad_(False)
        self.config = self.model.config
        self.config_identity_path = (config_identity_path if config_identity_path is not None
                                     else text_config.to_dict().get('_name_or_path', ''))
        self.contract = {
            "model_type": "qwen4_exp_text", "weights": weight_id,
            "config": config_identity(text_config, config_identity_path), "dtype": dtype,
            "backend": "transformers-reference-v1",
            "feature_policy": FEATURE_POLICY, "layer_ids": self.layer_ids,
            "hidden_size": text_config.hidden_size, "vocab_size": text_config.vocab_size,
        }

    @torch.inference_mode()
    def capture(self, input_ids, loss_mask):
        ids = torch.as_tensor(input_ids, dtype=torch.long).reshape(1, -1)
        mask = torch.as_tensor(loss_mask, dtype=torch.float32).reshape(-1)
        if ids.shape[1] != mask.numel() or ids.shape[1] < 1:
            raise ValueError("Token ids and loss mask must have equal length >= 1")
        if ids.min() < 0 or ids.max() >= self.config.vocab_size:
            raise ValueError("Token id outside target vocabulary")
        if ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("Sequence exceeds target position limit; split the data explicitly")
        states, handles = {}, []

        def hook(index):
            def capture_layer(_module, _args, output):
                # Raw outputs have width hc_count * H. SpecForge's stock DFlash2
                # projector expects H per selected layer. Apply the teacher's
                # learned final HC mixer, identically in training and verification.
                states[index] = self.model.model.hyper_connection_mixer(output).detach().cpu().clone()
            return capture_layer

        try:
            for index in self.layer_ids:
                handles.append(self.model.model.layers[index].register_forward_hook(hook(index)))
            out = self.model.model(input_ids=ids, use_cache=False, return_dict=True)
            features = {
                "input_ids": ids[0].cpu(), "loss_mask": mask.cpu(),
                "hidden_states": torch.cat([states[i] for i in self.layer_ids], dim=-1),
                "target_last_hidden_states": out.last_hidden_state.detach().cpu().clone(),
            }
            if any(not torch.isfinite(v).all() for v in features.values()):
                raise ValueError("Non-finite teacher features")
            return features
        finally:
            for handle in handles:
                handle.remove()

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens, **options):
        from .generation import generate
        return generate(self, input_ids, max_new_tokens, **options)

    @torch.inference_mode()
    def prefill(self, input_ids):
        output = self.model.model(input_ids=torch.tensor([input_ids], dtype=torch.long), use_cache=True)
        return output.past_key_values, output.last_hidden_state

    @torch.inference_mode()
    def decode(self, token, state):
        output = self.model.model(input_ids=torch.tensor([[token]], dtype=torch.long),
                                  past_key_values=state, use_cache=True)
        return output.past_key_values, output.last_hidden_state

    def save_assets(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file({
            "embed_tokens.weight": self.model.get_input_embeddings().weight.detach().cpu().contiguous(),
            "lm_head.weight": self.model.get_output_embeddings().weight.detach().cpu().contiguous(),
        }, directory / "teacher.safetensors")
        self.config.save_pretrained(directory)
        write_json(directory / "contract.json", self.contract)
        write_json(directory / 'origin.json', dict(config_identity_path=self.config_identity_path))

    def embed(self, ids):
        return self.model.get_input_embeddings()(torch.as_tensor(ids, dtype=torch.long)).float()

    def logits(self, hidden):
        return self.model.get_output_embeddings()(hidden.to(self.model.dtype)).float()

    def close(self):
        pass


def load_engine(model_path, layer_ids, *, backend="torch", devices="all", **options):
    if backend == "torch":
        if str(devices) not in ("all", "1"):
            raise ValueError("The torch reference backend supports one CPU device")
        return QwenEngine(model_path, layer_ids, **options)
    if backend == "jax":
        if str(devices) != "1":
            from .expert_parallel import ExpertParallelEngine
            return ExpertParallelEngine(model_path, layer_ids, devices=devices, **options)
        from .jax_engine import JaxQwenEngine
        return JaxQwenEngine(model_path, layer_ids, **options)
    raise ValueError(f"Unknown backend: {backend}")
