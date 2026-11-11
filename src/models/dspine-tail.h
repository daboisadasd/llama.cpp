// SPDX-License-Identifier: Apache-2.0
#pragma once

// First native DSpine greedy *proposal tail*, built from stock GGML ops:
//    learned final code write -> target BF16 head top-K candidates
//    -> exact-sender K-way score table -> small serial transition table.
// It can be compiled to the same GGML CPU/CUDA/RPC backends as llama.cpp.
//
// The trained 5-layer DFlash-derived attention backbone must still supply
// hidden/residual; TG/TSV are precomputed weight tables and MUST correspond
// to the same target BF16 output projection and checkpoint. Target verification
// and ReplaySSM are not implemented by this helper.

#include "dspine-bridge.h"
#include "dspine-refine.h"

namespace llama_dspine {

struct tail_result {
    bridge_result bridge;
    ggml_tensor * candidate_ids = nullptr; // [K,T-1] on the drafting backend
    ggml_tensor * predecessor_ids = nullptr; // [K,T-1] gathered from previous candidate row
    ggml_tensor * transitions = nullptr; // [K,T-1] previous-lane -> next-lane map
};

static inline tail_result build_greedy_tail(
        ggml_context * ctx,
        const bridge_inputs & input,
        const bridge_weights & learned,
        const refine_weights & refine,
        int block_size,
        int top_k,
        float norm_eps) {
    GGML_ASSERT(ctx && input.hidden && input.anchor_token_ids);
    GGML_ASSERT(input.hidden->ne[1] == block_size);
    GGML_ASSERT(block_size >= 2 && block_size <= 16);
    GGML_ASSERT(top_k >= 2 && top_k <= 64 && (top_k & (top_k-1)) == 0);
    auto bridge = build_bridge(ctx, input, learned, block_size, norm_eps);
    const int64_t h = input.hidden->ne[0];
    const int64_t gamma = block_size - 1;

    // Initial candidates are selected from post-last-write normalized
    // hidden states. The target LM head must be BF16 and identical to the
    // table-precomputation head, not the quantized target's Q4_K_M head.
    auto * proposal_rows = ggml_view_2d(ctx, bridge.normalized, h, gamma,
                                        bridge.normalized->nb[1],
                                        bridge.normalized->nb[1]);
    auto * logits = ggml_mul_mat(ctx, refine.lm_head, proposal_rows);
    auto * candidate_ids = ggml_top_k(ctx, logits, top_k);

    // Predecessor set: the first token can follow only the known anchor,
    // later tokens choose among the previous row's K candidates. Both
    // stages are computed on the same GGML backend -- no host roundtrip.
    auto * anchor = ggml_view_1d(ctx, input.anchor_token_ids, 1, 0);
    auto * anchor_shape = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, 1);
    auto * anchor_ids = ggml_repeat(ctx, anchor, anchor_shape);
    ggml_tensor * predecessor_ids = anchor_ids;
    if (gamma > 1) {
        auto * prior = ggml_view_2d(ctx, candidate_ids, top_k, gamma-1,
                                   candidate_ids->nb[1], 0);
        predecessor_ids = ggml_cont(ctx, ggml_concat(ctx, anchor_ids, prior, 1));
    }

    auto * transitions = build_serial_table(ctx,
        {bridge.raw, bridge.rms, bridge.packed, candidate_ids, predecessor_ids},
        refine, block_size, top_k);
    GGML_ASSERT(transitions->ne[0] == top_k && transitions->ne[1] == gamma);
    return {bridge, candidate_ids, predecessor_ids, transitions};
}

} // namespace llama_dspine
