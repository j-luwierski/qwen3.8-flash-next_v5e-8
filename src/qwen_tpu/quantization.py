"""Resumable, bounded-memory weight-only quantization for the streaming reader.

This is a project format, not HF's bitsandbytes/GGUF format. Matrices use
symmetric row groups of 64 values; norms and integer n-gram metadata stay exact.
The embedding and LM head retain their source dtype for exact training assets.
"""

import math
from pathlib import Path

import torch
from safetensors.torch import save

from .io import atomic_write, checkpoint_identity, digest, read_json, sha256, write_json


def quantize(value, bits, group_size):
    shape = value.shape
    padded = math.ceil(shape[-1] / group_size) * group_size
    values = torch.nn.functional.pad(value.float(), (0, padded - shape[-1])).reshape(*shape[:-1], -1, group_size)
    maximum = (1 << (bits - 1)) - 1
    scales = (values.abs().amax(-1) / maximum).clamp_min(1e-12)
    rounded = torch.round(values / scales.unsqueeze(-1)).clamp(-maximum, maximum).to(torch.int8)
    if bits == 4:
        unsigned = (rounded.to(torch.int16) + 8).to(torch.uint8).flatten(-2)
        packed = unsigned[..., ::2] | (unsigned[..., 1::2] << 4)
    else:
        packed = rounded.flatten(-2)
    return {'values': packed.contiguous(), 'scales': scales.contiguous()}


def dequantize(values, scales, bits, group_size, width):
    if bits == 4:
        integers = torch.stack((values & 15, values >> 4), dim=-1).flatten(-2).float() - 8
    else:
        integers = values.float()
    result = (integers.reshape(*scales.shape, group_size) * scales.unsqueeze(-1)).flatten(-2)
    return result[..., :width]


class QuantizedSlice:
    def __init__(self, reader, description):
        self.reader, self.description = reader, description

    def get_shape(self):
        return self.description['shape']

    def get_dtype(self):
        return self.description['dtype']

    def __getitem__(self, selection):
        shape = self.get_shape()
        if isinstance(selection, int):
            first, stop, scalar = selection, selection + 1, True
            if first < 0:
                first, stop = shape[0] + selection, shape[0] + selection + 1
        elif isinstance(selection, slice):
            first, stop, step = selection.indices(shape[0])
            if step != 1:
                raise ValueError('Only contiguous first-axis weight slices are supported')
            scalar = False
        else:
            raise ValueError('Unsupported quantized weight selection')
        if not 0 <= first <= stop <= shape[0]:
            raise IndexError('Quantized weight row outside tensor')
        pieces = []
        for part in self.description['parts']:
            a, b = max(first, part['start']), min(stop, part['stop'])
            if a >= b:
                continue
            file = self.reader._file(part['file'])
            selected = slice(a - part['start'], b - part['start'])
            if self.description['bits']:
                values = file.get_slice('values')[selected]
                scales = file.get_slice('scales')[selected]
                piece = dequantize(values, scales, self.description['bits'], self.description['group_size'], shape[-1])
            else:
                piece = file.get_slice('values')[selected]
            pieces.append(piece)
        if not pieces:
            return torch.empty((0, *shape[1:]), dtype=torch.float32)
        result = torch.cat(pieces, dim=0)
        return result[0] if scalar else result


def prepare_weights(model, out, *, bits=8, group_size=64, chunk_mib=32, control=None):
    from .runtime import check
    check(control, 'prepare_weights')
    from .jax_engine import TensorReader
    if bits not in (4, 8) or group_size < 2 or group_size % 2 or chunk_mib < 1:
        raise ValueError('bits must be 4/8, group_size positive even, chunk_mib positive')
    model, out = Path(model).resolve(), Path(out).resolve()
    if model == out or out.is_relative_to(model):
        raise ValueError('Quantized output must be outside the source checkpoint')
    if (model / 'qwen_quantization.json').exists():
        raise ValueError('Quantize the original checkpoint, not an already quantized export')
    identity = {'source_weights': checkpoint_identity(model), 'source_config': sha256(model / 'config.json'),
                'format': 'qwen-stream-quant-v1', 'bits': bits, 'group_size': group_size,
                'chunk_mib': chunk_mib, 'keep_exact': ['model.embed_tokens.weight', 'lm_head.weight']}
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'quantization.request.json').exists():
        if read_json(out / 'quantization.request.json') != identity:
            raise ValueError('Quantization resume options or source changed')
    elif any(out.iterdir()):
        raise ValueError('Quantization output is nonempty and belongs to another job')
    write_json(out / 'quantization.request.json', identity)
    descriptions, weight_map = {}, {}
    reader = TensorReader(model)
    try:
        for name in sorted(reader.mapping):
            view = reader.slice(name)
            shape, dtype = list(view.get_shape()), view.get_dtype()
            if not shape:
                raise ValueError(f'Scalar checkpoint tensor is unsupported: {name}')
            quantized = len(shape) >= 2 and dtype in ('BF16', 'F16', 'F32') and name not in identity['keep_exact']
            rows = max(1, int(chunk_mib * 1024 ** 2 / (math.prod(shape[1:]) * 4)))
            parts = []
            for start in range(0, shape[0], rows):
                check(control, 'prepare_weights')
                stop = min(start + rows, shape[0])
                key = digest([name, start, stop])
                filename = f'weights/{key}.safetensors'
                file, receipt = out / filename, out / 'receipts' / f'{key}.json'
                if receipt.exists():
                    if not file.exists() or sha256(file) != read_json(receipt)['sha256']:
                        raise ValueError('Corrupted completed quantization chunk')
                else:
                    value = view[start:stop]
                    if quantized:
                        stored = quantize(value, bits, group_size)
                    else:
                        stored = {'values': value}
                    atomic_write(file, lambda f: f.write(save(stored)))
                    write_json(receipt, {'sha256': sha256(file)})
                part = {'start': start, 'stop': stop, 'file': filename}
                parts.append(part)
                weight_map[f'{name}:{start}:{stop}'] = filename
            descriptions[name] = {'shape': shape, 'dtype': 'BF16' if quantized else dtype,
                                  'bits': bits if quantized else 0, 'group_size': group_size, 'parts': parts}
        # Metadata and tokenizer are small; no source model weights are copied.
        for path in model.iterdir():
            if path.is_file() and (path.suffix in ('.json', '.jinja', '.model', '.txt', '.tiktoken')) and path.name != 'model.safetensors.index.json':
                atomic_write(out / path.name, lambda f, path=path: f.write(path.read_bytes()))
        write_json(out / 'model.safetensors.index.json', {'weight_map': weight_map})
        write_json(out / 'qwen_quantization.json', {'identity': identity, 'tensors': descriptions})
        return {'source_bytes': sum(p.stat().st_size for p in model.glob('*.safetensors')),
                'output_bytes': sum((out / filename).stat().st_size for filename in set(weight_map.values())),
                'tensors': len(descriptions), 'bits': bits}
    finally:
        reader.close()
