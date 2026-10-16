"""Autoregressive sampling with a per-request RNG and recurrent target cache."""

import numpy as np
import torch
from .io import digest
from .runtime import SessionPaused


def generate(engine, input_ids, max_new_tokens, *, temperature=0.0, top_p=1.0,
             top_k=0, seed=42, min_new_tokens=0, stop_token_ids=None,
             progress=None, on_progress=None, control=None):
    ids = list(input_ids)
    if not ids or max_new_tokens < 1 or not 0 <= min_new_tokens <= max_new_tokens:
        raise ValueError("Nonempty prompt, positive max_new_tokens and valid min_new_tokens required")
    if temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("temperature >= 0, 0 < top_p <= 1, top_k >= 0 required")
    limit = min(engine.config.max_position_embeddings, getattr(engine, "max_tokens", engine.config.max_position_embeddings))
    if len(ids) + max_new_tokens > limit:
        raise ValueError(f"Prompt plus generation exceeds context limit {limit}")
    eos = engine.config.eos_token_id
    stops = set(stop_token_ids if stop_token_ids is not None else
                (eos if isinstance(eos, list) else ([] if eos is None else [eos])))
    rng = np.random.default_rng(seed)
    request = digest(dict(ids=ids, max_new_tokens=max_new_tokens, temperature=temperature,
                          top_p=top_p, top_k=top_k, seed=seed, min_new_tokens=min_new_tokens,
                          stops=sorted(stops)))
    prefix_length = len(ids)
    if progress:
        if progress['request'] != request or progress['input_ids'][:prefix_length] != ids:
            raise ValueError('Generation resume request differs')
        ids = list(progress['input_ids'])
        if not prefix_length <= len(ids) <= prefix_length + max_new_tokens:
            raise ValueError('Invalid generation resume length')
        rng.bit_generator.state = progress['rng_state']
        if progress['complete']:
            return ids
    last_saved = control.clock() if control else 0

    def save(complete=False):
        nonlocal last_saved
        if on_progress:
            on_progress(dict(request=request, input_ids=list(ids), rng_state=rng.bit_generator.state,
                             complete=complete))
        if control:
            last_saved = control.clock()

    def stop_if_due():
        if control and control.should_stop():
            save()
            raise SessionPaused('generation', control.reason)

    stop_if_due()
    state, hidden = engine.prefill(ids)
    for position in range(len(ids)-prefix_length, max_new_tokens):
        stop_if_due()
        logits = engine.logits(hidden[:, -1]).detach().float().numpy()[0].astype(np.float64)
        if position < min_new_tokens:
            for token in stops:
                logits[token] = -np.inf
        if not np.isfinite(logits).any():
            raise ValueError("No finite logits available for sampling")
        if temperature == 0:
            token = int(logits.argmax())
        else:
            logits /= temperature
            order = np.argsort(-logits, kind="stable")
            if top_k:
                order = order[:top_k]
            values = logits[order]
            probs = np.exp(values - values.max())
            probs /= probs.sum()
            keep = np.cumsum(probs) - probs < top_p
            order, probs = order[keep], probs[keep]
            probs /= probs.sum()
            token = int(rng.choice(order, p=probs))
        ids.append(token)
        if token in stops or position + 1 == max_new_tokens:
            save(complete=True)
            break
        if control and control.due(last_saved):
            save()
        stop_if_due()
        state, hidden = engine.decode(token, state)
    return ids
