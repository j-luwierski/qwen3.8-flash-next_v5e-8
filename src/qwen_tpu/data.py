"""Prompt ingestion, exact chat-template identities and on-policy examples."""

import json
from pathlib import Path

from transformers import AutoTokenizer

from .io import atomic_write, digest, read_json, sha256, write_json


def normalize_prompt(raw):
    messages = raw.get("messages", raw.get("conversations"))
    if messages is None:
        text = raw.get("prompt", raw.get("input", raw.get("text")))
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Expected prompt/input/text or a messages/conversations list")
        messages = [{"role": "user", "content": text}]
    normalized = []
    roles = {"human": "user", "gpt": "assistant"}
    for message in messages:
        role = roles.get(message.get("from"), message.get("role", message.get("from")))
        content = message.get("content", message.get("value", ""))
        if role not in ("system", "user", "assistant", "tool") or not isinstance(content, str):
            raise ValueError("Only text chat messages with system/user/assistant/tool roles are supported")
        normalized.append({**{k: v for k, v in message.items() if k in ("tool_calls", "tool_call_id", "name")},
                           "role": role, "content": content})
    # Earlier turns are inference context. Only the final assistant response is
    # regenerated and supervised, so tool results in that context remain valid.
    if normalized and normalized[-1]["role"] == "assistant":
        normalized.pop()
    if not normalized or normalized[-1]["role"] not in ("user", "tool"):
        raise ValueError("A prompt must end in a user message or tool result")
    result = {"messages": normalized}
    if "tools" in raw:
        result["tools"] = raw["tools"]
    result["id"] = str(raw.get("id", digest(result)))
    return result


def fetch_prompts(out, *, dataset=None, revision=None, data=None, samples=5000, seed=42,
                  split="train", dataset_config=None, control=None):
    """Resume local files at a byte offset, HF streams at a deterministic row cursor."""
    from .runtime import check
    check(control, 'fetch_prompts')
    if (dataset is None) == (data is None) or samples < 1:
        raise ValueError("Choose exactly one of dataset/data and a positive sample count")
    out = Path(out)
    request = dict(dataset=dataset, revision=revision, samples=samples, seed=seed, split=split,
                   dataset_config=dataset_config, local_sha256=sha256(data) if data else None)
    state_path = out.with_suffix('.fetch.json')
    state = read_json(state_path) if state_path.exists() else None
    if state and state['request'] != request:
        raise ValueError('Prompt fetch inputs changed during resume')
    if out.exists():
        if state and state.get('complete') and control:
            if sha256(out) != read_json(out.with_suffix('.source.json'))['sha256']:
                raise ValueError('Completed prompts checksum mismatch')
            return state['selected']
        raise ValueError(f"Prompt output already exists: {out}")
    if dataset:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        pinned = state['source']['revision'] if state else HfApi().dataset_info(dataset, revision=revision or 'main').sha
        rows = load_dataset(dataset, name=dataset_config, split=split, revision=pinned, streaming=True)
        rows = rows.shuffle(seed=seed, buffer_size=min(10000, max(samples, 1000)))
        position = state['position'] if state else 0
        rows = ((row, index+1) for index, row in enumerate(rows.skip(position), start=position))
        source = dict(dataset=dataset, revision=pinned, seed=seed, split=split, dataset_config=dataset_config)
    else:
        def local_rows():
            with open(data, 'rb') as file:
                file.seek(state['position'] if state else 0)
                while line := file.readline():
                    if line.strip():
                        yield json.loads(line), file.tell()
        rows = local_rows()
        source = dict(file_sha256=request['local_sha256'])
    selected = list(state['selected']) if state else []
    seen = {row['id'] for row in selected}
    position = state['position'] if state else 0
    def save(complete=False):
        write_json(state_path, dict(request=request, source=source, selected=selected,
                                   position=position, complete=complete))
        if control:
            control.saved()
    try:
        if len(selected) < samples:
            for row, next_position in rows:
                check(control, 'fetch_prompts')
                try:
                    prompt = normalize_prompt(row)
                    key = digest({k: v for k, v in prompt.items() if k != 'id'})
                    if key not in seen:
                        seen.add(key)
                        prompt['id'] = key
                        selected.append(prompt)
                except ValueError:
                    pass
                position = next_position
                if control and control.due():
                    save()
                if len(selected) == samples:
                    break
    except BaseException:
        save()
        raise
    if not selected:
        raise ValueError('No valid unique prompts found')
    atomic_write(out, lambda f: f.write(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected).encode()))
    write_json(out.with_suffix('.source.json'), {**source, 'samples': len(selected), 'sha256': sha256(out)})
    save(complete=True)
    return selected


class ChatTokenizer:
    def __init__(self, path, revision=None, template_kwargs=None):
        if not Path(path).is_dir() and (revision is None or len(revision) != 40):
            raise ValueError("Remote tokenizer requires an immutable 40-character revision")
        self.tokenizer = AutoTokenizer.from_pretrained(path, revision=revision, trust_remote_code=False)
        self.kwargs = dict(template_kwargs or {})
        if not self.tokenizer.chat_template:
            raise ValueError("Tokenizer has no chat template")
        self.identity = digest({"tokenizer": self.tokenizer.backend_tokenizer.to_str(),
                                "template": self.tokenizer.chat_template,
                                "special_tokens": self.tokenizer.special_tokens_map,
                                "kwargs": self.kwargs})

    def prefix(self, prompt):
        ids = self.tokenizer.apply_chat_template(prompt["messages"], tools=prompt.get("tools"),
                                                 tokenize=True, return_dict=False, add_generation_prompt=True, **self.kwargs)
        return list(ids)

    def example(self, engine, prompt, *, max_length, generation, seed,
                progress=None, on_progress=None, control=None):
        from .generation import generate
        prefix = self.prefix(prompt)
        if len(prefix) + 2 > max_length:
            return {"status": "skipped", "reason": "prompt_exceeds_context_budget", "prompt_tokens": len(prefix)}
        options = dict(generation)
        max_new = min(options.pop("max_new_tokens", 512), max_length - len(prefix))
        if max_new < 2:
            raise ValueError("Training generation requires max_new_tokens >= 2")
        ids = generate(engine, prefix, max_new, seed=seed, min_new_tokens=1,
                       progress=progress, on_progress=on_progress, control=control, **options)
        if len(ids) - len(prefix) < 2:
            return {"status": "skipped", "reason": "response_has_fewer_than_two_tokens"}
        response = ids[len(prefix):]
        return {"status": "complete", "id": prompt["id"], "input_ids": ids,
                "loss_mask": [0] * len(prefix) + [1] * len(response),
                "messages": prompt["messages"],
                "assistant_raw": self.tokenizer.decode(response, skip_special_tokens=False),
                "prompt_raw": self.tokenizer.decode(prefix, skip_special_tokens=False),
                "prompt_tokens": len(prefix), "response_tokens": len(response),
                "finish_reason": "length" if len(response) == max_new else "stop"}
