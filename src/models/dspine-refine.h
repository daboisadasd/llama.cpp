// SPDX-License-Identifier: Apache-2.0
#pragma once

// DSpine's exact-sender greedy serial refinement over a proposed K-way
// candidate lattice. Implemented using stock GGML ops (CPU, CUDA, RPC).
//
// NOT a complete DSpine decoder. Candidate IDs and per-position predecessor
// candidate IDs must be produced by the draft backbone/first-stage top-K,
// and no sampling/rejection path exists here yet.
//
// This follows Tencent DSpine's serial_table / serial_chain formulation:
//   score(prev, curr) = lm_head[curr] * (raw * output_norm)
//                     + rms * (TG[curr] dot
//                        (sigmoid(query + TSV_g[prev] + gate_bias)
//                         * TSV_value[prev]))
// The output [K, blocks*(T-1)] is the index of the best candidate for EACH
// predecessor lane, not simply argmax of draft logits.

#include "ggml.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace llama_dspine {

struct refine_weights {
    ggml_tensor * lm_head = nullptr;  // [H,V] BF16 (target BF16 head)
    ggml_tensor * norm_w  = nullptr;  // [H]   BF16
    ggml_tensor * tg      = nullptr;  // [R,V] BF16 (precomputed sender/head table)
    ggml_tensor * tsv     = nullptr;  // [2R,V] BF16 (precomputed codebook table)
    ggml_tensor * gate_bias = nullptr; // [R] BF16
};

struct refine_inputs {
    ggml_tensor * raw = nullptr;     // [H,blocks*T] BF16 (pre-write bridge)
    ggml_tensor * rms = nullptr;     // [1,blocks*T] F32
    ggml_tensor * packed = nullptr;  // [3R,blocks*T] BF16
    ggml_tensor * candidate_ids = nullptr; // [K,blocks*(T-1)] I32
    ggml_tensor * previous_ids  = nullptr; // [K,blocks*(T-1)] I32
};

// All tensor data stays on the selected GGML backend until the final Kx(T-1)
// integer transition table is returned. A host may walk this tiny table with
// serial_chain without a separate RPC command for each proposed token.
static inline ggml_tensor * build_serial_table(
        ggml_context * ctx,
        const refine_inputs & in,
        const refine_weights & w,
        int64_t block_size,
        int64_t k) {
    GGML_ASSERT(ctx && in.raw && in.rms && in.packed && in.candidate_ids && in.previous_ids);
    GGML_ASSERT(w.lm_head && w.norm_w && w.tg && w.tsv && w.gate_bias);
    GGML_ASSERT(2 <= block_size && block_size <= 16);
    GGML_ASSERT(k >= 2 && k <= 64 && (k & (k-1)) == 0);
    const int64_t h = in.raw->ne[0];
    const int64_t n = in.raw->ne[1];
    const int64_t r = w.gate_bias->ne[0];
    const int64_t n_blocks = n / block_size;
    const int64_t steps = n_blocks * (block_size - 1);
    GGML_ASSERT(n_blocks >= 1 && n_blocks <= 8 && n == n_blocks*block_size);
    GGML_ASSERT(w.norm_w->ne[0] == h && w.lm_head->ne[0] == h);
    GGML_ASSERT(w.tg->ne[0] == r && w.tg->ne[1] == w.lm_head->ne[1]);
    GGML_ASSERT(w.tsv->ne[0] == 2*r && w.tsv->ne[1] == w.lm_head->ne[1]);
    GGML_ASSERT(in.packed->ne[0] == 3*r && in.packed->ne[1] == n);
    GGML_ASSERT(in.rms->type == GGML_TYPE_F32 && in.rms->ne[0] == 1 && in.rms->ne[1] == n);
    GGML_ASSERT(in.candidate_ids->type == GGML_TYPE_I32 && in.previous_ids->type == GGML_TYPE_I32);
    GGML_ASSERT(in.candidate_ids->ne[0] == k && in.candidate_ids->ne[1] == steps);
    GGML_ASSERT(ggml_are_same_shape(in.candidate_ids, in.previous_ids));
    ggml_tensor * table = nullptr;

    auto * norm_w = ggml_cast(ctx, w.norm_w, GGML_TYPE_F32);
    auto * bias = ggml_cast(ctx, w.gate_bias, GGML_TYPE_F32);
    for (int64_t step = 0; step < steps; step++) {
        const int64_t pos = (step / (block_size-1))*block_size + (step%(block_size-1)) + 1;
        const size_t cand_off = (size_t)step * in.candidate_ids->nb[1];
        const size_t prev_off = (size_t)step * in.previous_ids->nb[1];
        auto * cand = ggml_view_1d(ctx, in.candidate_ids, k, cand_off);
        auto * prev = ggml_view_1d(ctx, in.previous_ids, k, prev_off);

        auto * raw = ggml_view_2d(ctx, in.raw, h, 1, in.raw->nb[1], (size_t)pos * in.raw->nb[1]);
        raw = ggml_cast(ctx, raw, GGML_TYPE_F32);
        // p[y] = sum_h BF16(lm_head[y,h]) * (raw[h] * norm_w[h])
        auto * normed = ggml_mul(ctx, raw, ggml_repeat(ctx, norm_w, raw));
        auto * candidate_head = ggml_cast(ctx, ggml_get_rows(ctx, w.lm_head, cand), GGML_TYPE_F32);
        auto * base = ggml_mul_mat(ctx, candidate_head, normed);  // [K,1]

        auto * candidate_tg = ggml_cast(ctx, ggml_get_rows(ctx, w.tg, cand), GGML_TYPE_F32); // [R,K]
        auto * previous_tsv = ggml_cast(ctx, ggml_get_rows(ctx, w.tsv, prev), GGML_TYPE_F32); // [2R,K]
        auto * prev_gate  = ggml_view_2d(ctx, previous_tsv, r, k, previous_tsv->nb[1], 0);
        auto * prev_value = ggml_view_2d(ctx, previous_tsv, r, k, previous_tsv->nb[1],
                                          (size_t)r * previous_tsv->nb[0]);
        auto * query = ggml_view_2d(ctx, in.packed, r, 1, in.packed->nb[1],
                                   (size_t)pos * in.packed->nb[1]);
        query = ggml_cast(ctx, query, GGML_TYPE_F32);

        auto * gate_pre = ggml_add(ctx,
            ggml_add(ctx, prev_gate, ggml_repeat(ctx, query, prev_gate)),
            ggml_repeat(ctx, bias, prev_gate));
        auto * message = ggml_mul(ctx, ggml_sigmoid(ctx, gate_pre), prev_value);
        auto * adjust = ggml_mul_mat(ctx, candidate_tg, message); // [K,K], prev index in dimension 1
        auto * rms = ggml_view_1d(ctx, in.rms, 1, (size_t)pos * in.rms->nb[1]);
        adjust = ggml_mul(ctx, adjust, ggml_repeat(ctx, rms, adjust));
        auto * score = ggml_add(ctx, ggml_repeat(ctx, base, adjust), adjust);
        auto * best = ggml_argmax(ctx, score); // [1,K] transition lanes
        table = table ? ggml_concat(ctx, table, best, 1) : best;
    }
    // ggml_argmax returns a *vector* of length K, not a [1,K] matrix.
    // Concatenating those vectors on dim 1 naturally builds [K,steps].
    // A reshape would hide the real layout and can break the transition walk.
    GGML_ASSERT(table != nullptr && table->ne[0] == k && table->ne[1] == steps);
    return ggml_cont(ctx, table);
}

static inline std::vector<int32_t> serial_chain(
        const std::vector<int32_t> & transition,
        const std::vector<int32_t> & candidates,
        int64_t block_size,
        int64_t k) {
    if (block_size < 2 || k < 2 || transition.size() != candidates.size() ||
            transition.size() % (size_t)((block_size-1)*k) != 0) {
        throw std::invalid_argument("invalid DSpine K-way serial chain shape");
    }
    const size_t steps = transition.size() / (size_t)k;
    std::vector<int32_t> output;
    output.reserve(steps);
    for (size_t begin=0; begin<steps; begin += (size_t)(block_size-1)) {
        int32_t j = transition[begin*k];
        for (size_t row=begin; row<begin+(size_t)(block_size-1); row++) {
            if (j < 0 || j >= k) throw std::runtime_error("DSpine transition index out of range");
            output.push_back(candidates[row*k + (size_t)j]);
            if (row+1<begin+(size_t)(block_size-1)) {
                j = transition[(row+1)*k + (size_t)j];
            }
        }
    }
    return output;
}

} // namespace llama_dspine
