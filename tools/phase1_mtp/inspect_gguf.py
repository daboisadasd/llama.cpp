#!/usr/bin/env python3
"""Read the GGUF header only; never loads or maps model tensor payloads."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import struct

_TYPES = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f',
          7: '?', 10: 'Q', 11: 'q', 12: 'd'}

# GGML tensor type enum IDs (different from GGUF metadata value types).
# See ggml/include/ggml.h. Unknown types are kept as TYPE_<number>.
_TENSOR_TYPES = {0: 'F32', 1: 'F16', 2: 'Q4_0', 8: 'Q8_0', 30: 'BF16'}


class GGUFError(ValueError):
    pass


class Reader:
    def __init__(self, f):
        self.f = f
        self.version = None
        # Save file size so bulk metadata skipping fails safely on truncated headers.
        here = f.tell()
        self.file_size = f.seek(0, os.SEEK_END)
        f.seek(here)

    def skip(self, n):
        if n < 0 or self.f.tell() + n > self.file_size:
            raise GGUFError('truncated GGUF metadata while skipping array payload')
        self.f.seek(n, os.SEEK_CUR)

    def skip_value(self, tag, depth=0):
        # Avoid allocating hundreds of thousands of tokenizer vocabulary entries.
        if depth > 4:
            raise GGUFError('nested metadata arrays too deep')
        if tag in _TYPES:
            self.skip(struct.calcsize('<' + _TYPES[tag]))
        elif tag == 8:
            self.skip(self.unpack('Q'))
        elif tag == 9:
            element_type, count = self.unpack('I'), self.unpack('Q')
            self.skip_array(element_type, count, depth + 1)
        else:
            raise GGUFError(f'unknown GGUF value type {tag}')

    def skip_array(self, element_type, count, depth=0):
        if count > 1000000:
            raise GGUFError('unreasonable array length')
        if element_type in _TYPES:
            self.skip(struct.calcsize('<' + _TYPES[element_type]) * count)
        elif element_type in (8, 9):
            for _ in range(count):
                self.skip_value(element_type, depth)
        else:
            raise GGUFError(f'unknown GGUF array element type {element_type}')

    def read(self, n):
        if n < 0 or n > 64 * 1024 * 1024:
            raise GGUFError('unreasonable field size')
        b = self.f.read(n)
        if len(b) != n:
            raise GGUFError('truncated GGUF metadata')
        return b

    def unpack(self, fmt):
        fmt = '<' + fmt
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self):
        n = self.unpack('Q')
        if n > 1024 * 1024:
            raise GGUFError('unreasonable string length')
        return self.read(n).decode('utf-8', 'replace')

    def value(self, tag, depth=0):
        if depth > 4:
            raise GGUFError('nested metadata arrays too deep')
        if tag in _TYPES:
            return self.unpack(_TYPES[tag])
        if tag == 8:
            return self.string()
        if tag == 9:
            tag2 = self.unpack('I')
            n = self.unpack('Q')
            if n > 1000000:
                raise GGUFError('unreasonable array length')
            if n > 256:
                # GGUF tokenizer metadata commonly contains large STRING arrays.
                # Skip all element kinds without materializing the vocabulary.
                self.skip_array(tag2, n, depth + 1)
                return {'count': n, 'value_type': tag2}
            return [self.value(tag2, depth + 1) for _ in range(n)]
        raise GGUFError(f'unknown GGUF value type {tag}')


def inspect(path: Path):
    with path.open('rb') as f:
        r = Reader(f)
        if r.read(4) != b'GGUF':
            raise GGUFError(f'{path} is not a GGUF file')
        version = r.unpack('I')
        if version not in (2, 3):
            raise GGUFError(f'unsupported GGUF version {version}')
        num_tensors, n_meta = r.unpack('Q'), r.unpack('Q')
        if num_tensors > 100000 or n_meta > 100000:
            raise GGUFError('unreasonable tensor or metadata count')
        metadata = {}
        for _ in range(n_meta):
            key = r.string()
            value = r.value(r.unpack('I'))
            if key.endswith(('architecture', 'name', 'nextn_predict_layers', 'embedding_length', 'block_count', 'file_type', 'context_length', 'tokenizer.ggml.model')) or 'nextn' in key or key == 'general.architecture':
                metadata[key] = value
        names = []
        mtp_names = []
        tensor_type_counts = {}
        for _ in range(num_tensors):
            name = r.string()
            names.append(name)
            if re.search(r'(^nextn\.|\.nextn\.|^blk\.6[4-9]\.|mtp)', name, re.I):
                mtp_names.append(name)
            dims = r.unpack('I')
            if dims > 8:
                raise GGUFError('unreasonable tensor dimension count')
            r.read(8 * dims)  # dimensions
            tensor_type_id = r.unpack('I')  # GGML type code
            r.read(8)  # tensor data offset
            type_name = _TENSOR_TYPES.get(tensor_type_id, f'TYPE_{tensor_type_id}')
            tensor_type_counts[type_name] = tensor_type_counts.get(type_name, 0) + 1
        return {
            'path': str(path.resolve()),
            'bytes': path.stat().st_size,
            'gguf_version': version,
            'tensor_count': num_tensors,
            'metadata': metadata,
            'mtp_tensor_count': len(mtp_names),
            'tensor_type_counts': tensor_type_counts,
            'mtp_tensor_examples': mtp_names[:24],
            'first_tensor_names': names[:12],
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('files', nargs='+', type=Path)
    args = p.parse_args()
    errors = 0
    for path in args.files:
        try:
            print(json.dumps(inspect(path), indent=2))
        except Exception as e:
            errors += 1
            print(f'{path}: ERROR: {e}', file=__import__('sys').stderr)
    raise SystemExit(bool(errors))

if __name__ == '__main__':
    main()
