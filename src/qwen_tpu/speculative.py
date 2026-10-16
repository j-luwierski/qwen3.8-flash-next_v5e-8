"""Correctness-first greedy DFlash2 verification for recurrent Flash-Next.

No cache cropping: DeltaNet and PLE carry recurrent/convolutional states that
cannot be rolled back by slicing attention K/V alone. Re-prefilling the accepted
prefix is slow, but avoids state contamination after rejected proposals.
"""

from pathlib import Path

import torch
from transformers import Qwen3Config

from .io import read_json


@torch.inference_mode()
def greedy_speculate(engine, draft_path, input_ids, max_new_tokens):
    from specforge.modeling.draft.dflash2 import DFlash2DraftModel

    draft_path = Path(draft_path)
    contract = read_json(draft_path / "target_contract.json")
    if any(contract[k] != v for k, v in engine.contract.items()):
        raise ValueError("Draft was trained with a different target/capture contract")
    config = Qwen3Config.from_pretrained(draft_path)
    draft = DFlash2DraftModel.from_pretrained(draft_path, config=config).eval()
    tokens, accepted_lengths = list(input_ids), []
    limit = len(tokens) + max_new_tokens
    if limit > min(engine.config.max_position_embeddings, getattr(engine, "max_tokens", engine.config.max_position_embeddings)):
        raise ValueError("Requested generation exceeds backend context limit")
    eos = engine.config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    if not tokens or max_new_tokens < 1:
        raise ValueError("A nonempty prompt and positive max_new_tokens are required")
    while len(tokens) < limit:
        prefix = engine.capture(tokens, [1] * len(tokens))
        first = int(engine.logits(prefix["target_last_hidden_states"][:, -1]).argmax(-1).item())
        if first in stops or len(tokens) + 1 == limit:
            return tokens + [first], accepted_lengths + [1]
        proposal = torch.full((1, draft.block_size), draft.mask_token_id, dtype=torch.long)
        proposal[0, 0] = first
        n = len(tokens)
        noise = engine.embed(proposal)
        # Context keys precede the draft's block keys; within-block attention is
        # bidirectional, while the full target remains causal for verification.
        positions = torch.arange(n + draft.block_size).unsqueeze(0)
        hidden = draft(position_ids=positions, target_hidden=prefix["hidden_states"].float(),
                       noise_embedding=noise, use_cache=False, is_causal=False)
        logits = draft.transform_unary_logits(engine.logits(hidden[:, 1:]))
        candidate_logits, candidate_ids = logits.topk(draft.candidate_selector.top_k, dim=-1)
        selected = draft.candidate_selector.greedy_path(
            candidate_ids=candidate_ids, unary_logits=candidate_logits,
            hidden_states=hidden[:, 1:], anchor_token_ids=proposal[:, 0])
        proposal[:, 1:] = selected
        verify_width = min(draft.block_size, limit - n)
        verify_ids = tokens + proposal[0, :verify_width].tolist()
        verified = engine.capture(verify_ids, [1] * len(verify_ids))
        posterior = engine.logits(verified["target_last_hidden_states"][:, n:])[0].argmax(-1).tolist()
        count = 1
        for j in range(1, verify_width):
            if int(proposal[0, j]) != posterior[j - 1]:
                break
            count += 1
        extension = proposal[0, :count].tolist() + [posterior[count - 1]]
        accepted_lengths.append(count)
        for token in extension:
            if len(tokens) == limit:
                break
            tokens.append(token)
            if token in stops:
                return tokens, accepted_lengths
    return tokens, accepted_lengths
