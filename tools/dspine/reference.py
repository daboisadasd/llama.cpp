#!/usr/bin/env python3
"""Tiny scalar DSpine adjacent-code reference for port tests, NOT an inference engine.

Formulas are independently transcribed from Tencent's
sglang_dspine/kernels/dspine.py (prepare_read/feature_message/feature_finish,
serial_chain). These functions deliberately do not implement BF16 rounding
or the target model; they are useful to check the graph semantics in isolation.
"""
from __future__ import annotations
import math


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def adjacent_message(packed, gate_bias, block_size):
    """packed[row] = [receiver R, sender R, value R] floats, T whole rows."""
    r = len(gate_bias)
    if block_size < 2 or not packed or len(packed) % block_size:
        raise ValueError("complete DSpine blocks required")
    if any(len(row) != 3 * r for row in packed):
        raise ValueError("packed row width must be 3 * rank")
    result = []
    for i, row in enumerate(packed):
        if i % block_size == 0:
            result.append([0.0] * r)
            continue
        prev = packed[i - 1]
        result.append([sigmoid(row[k] + prev[r+k] + gate_bias[k]) * prev[2*r+k]
                       for k in range(r)])
    return result


def adjacent_write(raw, rms, message, write_weight, block_size, norm_weight=None, eps=1e-6):
    """Apply the learned independent bridge write; returns residual and optional RMSNorm.

    raw: [n,H], rms: [n], message: [n,R], write_weight: [H,R].
    This does NOT implement BF16 casts and is not a replacement for CUDA kernels.
    """
    if block_size < 2 or not raw or len(raw) % block_size or len(raw) != len(rms) or len(raw) != len(message):
        raise ValueError("rows must form complete blocks with one RMS and message per row")
    h = len(raw[0]); r = len(message[0]);
    if len(write_weight) != h or any(len(w) != r for w in write_weight):
        raise ValueError("write projection shape must be [hidden, rank]")
    if any(len(x) != h for x in raw) or any(len(m) != r for m in message):
        raise ValueError("nonuniform row dimensions")
    if norm_weight is not None and len(norm_weight) != h:
        raise ValueError("norm weight width != hidden")
    output = []; normalized = []
    for i, row in enumerate(raw):
        scale = rms[i] if i % block_size else 0.0
        delta = [sum(write_weight[col][k] * message[i][k] for k in range(r))
                 for col in range(h)]
        value = [row[col] + scale * delta[col] for col in range(h)]
        output.append(value)
        if norm_weight is not None:
            inv = 1.0 / math.sqrt(sum(x*x for x in value) / h + eps)
            normalized.append([value[j] * inv * norm_weight[j] for j in range(h)])
    return output, normalized if norm_weight is not None else None


def serial_chain(index_rows, candidate_rows, block_size):
    """Follow row-wise 16-way transition lanes, matching DSpine greedy serial_chain."""
    if block_size < 2 or not index_rows or len(index_rows) != len(candidate_rows) or len(index_rows) % (block_size-1):
        raise ValueError("need block_size - 1 refinement rows per block")
    out = []
    for start in range(0, len(index_rows), block_size-1):
        k = len(candidate_rows[start])
        if k == 0 or any(len(ids) != k for ids in candidate_rows[start:start+block_size-1]):
            raise ValueError("candidate table shape mismatch")
        j = index_rows[start][0]
        for row in range(start, start+block_size-1):
            if not 0 <= j < k:
                raise ValueError("refinement index outside candidate lanes")
            out.append(candidate_rows[row][j])
            if row + 1 < start + block_size - 1:
                j = index_rows[row+1][j]
    return out
