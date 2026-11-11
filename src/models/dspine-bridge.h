// SPDX-License-Identifier: Apache-2.0
#pragma once

// DSpine's learned adjacent-code write, expressed in ordinary GGML graph ops.
// The same graph can later be placed on a GGML CUDA or RPC backend.
//
// Port milestone: this is an independent graph primitive, not yet wired to
// llama_model_dflash::graph or common_speculative_impl. DSpine is NOT runnable
// until the 5-layer drafter, serial K=16 refinement and verification are wired.
//
// Shapes use GGML's [ne0, ne1] order. All rows must form full T-token blocks.
// For each row, caller supplies:
//   anchor_token_ids [N] I32   (only rows 0, T, 2T,... are consulted),
//   predecessor_ids [N] I32   (0 for anchor; row - 1 otherwise),
//   anchor_mask [1,N] F32     (1 on anchor, otherwise 0),
//   edge_mask   [1,N] F32     (0 on anchor, otherwise 1).
// The inputs are intentionally explicit so block boundaries never leak across
// batches. GGML inputs/scalars are supplied by the model graph owner.

#include "ggml.h"
#include <cmath>
#include <cstdint>

namespace llama_dspine {

struct bridge_weights {
    ggml_tensor * codebook   = nullptr; // [C,V], BF16
    ggml_tensor * read_w     = nullptr; // [H,C], BF16
    ggml_tensor * gate_w     = nullptr; // [C,R], BF16
    ggml_tensor * sender_w   = nullptr; // [C,R], BF16 (predecessor_gate)
    ggml_tensor * value_w    = nullptr; // [C,R], BF16 (prev)
    ggml_tensor * gate_bias  = nullptr; // [R],   BF16
    ggml_tensor * write_w    = nullptr; // [R,H], BF16
    ggml_tensor * next_norm_w = nullptr; // [H],   BF16
};

struct bridge_inputs {
    ggml_tensor * hidden          = nullptr; // [H,N], BF16
    ggml_tensor * residual        = nullptr; // [H,N], optional BF16
    ggml_tensor * anchor_token_ids = nullptr; // [N] I32
    ggml_tensor * predecessor_ids = nullptr; // [N] I32
    ggml_tensor * anchor_mask     = nullptr; // [1,N] F32
    ggml_tensor * edge_mask       = nullptr; // [1,N] F32
    ggml_tensor * eps             = nullptr; // [1]   F32, checkpoint bridge.eps
};

struct bridge_result {
    ggml_tensor * raw        = nullptr; // [H,N], pre-write BF16 residual
    ggml_tensor * rms        = nullptr; // [1,N] F32 of raw
    ggml_tensor * packed     = nullptr; // [3R,N], receiver/sender/value
    ggml_tensor * residual   = nullptr; // [H,N], post-write BF16
    ggml_tensor * normalized = nullptr; // [H,N], next-layer RMSNorm(BF16)
};

// Build only. No tensors are allocated or executed here; the llama.cpp
// backend scheduler owns allocation and dispatch. All matmuls remain GGML ops.
// The exact Tencent BF16 rounding behavior still requires numeric comparison.
static inline bridge_result build_bridge(
        ggml_context * ctx,
        const bridge_inputs & in,
        const bridge_weights & w,
        int block_size,
        float norm_eps) {
    GGML_ASSERT(ctx && in.hidden && in.anchor_token_ids && in.predecessor_ids);
    GGML_ASSERT(in.anchor_mask && in.edge_mask && in.eps);
    GGML_ASSERT(w.codebook && w.read_w && w.gate_w && w.sender_w && w.value_w);
    GGML_ASSERT(w.gate_bias && w.write_w && w.next_norm_w);
    GGML_ASSERT(block_size >= 2 && norm_eps > 0);
    const int64_t h = in.hidden->ne[0];
    const int64_t n = in.hidden->ne[1];
    const int64_t c = w.codebook->ne[0];
    const int64_t r = w.gate_bias->ne[0];
    GGML_ASSERT(n > 0 && n % block_size == 0);
    GGML_ASSERT(in.hidden->type == GGML_TYPE_BF16 || in.hidden->type == GGML_TYPE_F32);
    GGML_ASSERT(in.eps->type == GGML_TYPE_F32 && ggml_nelements(in.eps) == 1);
    GGML_ASSERT(in.anchor_token_ids->type == GGML_TYPE_I32 && in.anchor_token_ids->ne[0] == n);
    GGML_ASSERT(in.predecessor_ids->type == GGML_TYPE_I32 && in.predecessor_ids->ne[0] == n);
    GGML_ASSERT(in.anchor_mask->type == GGML_TYPE_F32 && in.anchor_mask->ne[0] == 1 && in.anchor_mask->ne[1] == n);
    GGML_ASSERT(in.edge_mask->type == GGML_TYPE_F32 && in.edge_mask->ne[0] == 1 && in.edge_mask->ne[1] == n);
    GGML_ASSERT(w.read_w->ne[0] == h && w.read_w->ne[1] == c);
    GGML_ASSERT(w.codebook->ne[1] > 0);
    GGML_ASSERT(w.gate_w->ne[0] == c && w.gate_w->ne[1] == r);
    GGML_ASSERT(w.sender_w->ne[0] == c && w.sender_w->ne[1] == r);
    GGML_ASSERT(w.value_w->ne[0] == c && w.value_w->ne[1] == r);
    GGML_ASSERT(w.write_w->ne[0] == r && w.write_w->ne[1] == h);
    GGML_ASSERT(w.next_norm_w->ne[0] == h);
    if (in.residual) {
        GGML_ASSERT(ggml_are_same_shape(in.hidden, in.residual));
    }

    auto round_state = [&](ggml_tensor * t) {
        // Match SGLang's BF16 activation boundaries without throwing away
        // the F32 CPU graph reference path.
        return in.hidden->type == GGML_TYPE_BF16 ? ggml_cast(ctx, t, GGML_TYPE_BF16) : t;
    };
    auto * raw = in.residual ? round_state(ggml_add(ctx, in.hidden, in.residual)) : in.hidden;
    auto * raw_f32 = ggml_cast(ctx, raw, GGML_TYPE_F32);

    // Tencent prepare_read: RMS = sqrt(max(mean(raw^2), minimum-positive-f32)).
    // normalized = raw / (RMS + bridge-eps), rather than sqrt(mean+eps).
    auto * rms = ggml_sqrt(ctx, ggml_clamp(ctx,
                       ggml_mean(ctx, ggml_sqr(ctx, raw_f32)), 1.17549435e-38f, 1e30f));
    auto * read_input = round_state(ggml_div(ctx, raw_f32, ggml_add(ctx, rms, in.eps)));
    auto * projected = round_state(ggml_mul_mat(ctx, w.read_w, read_input));

    // Unit-normalized feature code at non-anchor rows.
    auto * projected_f32 = ggml_cast(ctx, projected, GGML_TYPE_F32);
    auto * code_length = ggml_sqrt(ctx,
        ggml_scale(ctx, ggml_mean(ctx, ggml_sqr(ctx, projected_f32)), (float)c));
    auto * normalized_code = ggml_scale(ctx,
        ggml_div(ctx, projected_f32, ggml_add(ctx, code_length, in.eps)), std::sqrt((float)c));

    // Anchor row bypasses the read projection: sqrt(C) * codebook[token].
    auto * anchor_code = ggml_scale(ctx, ggml_get_rows(ctx, w.codebook, in.anchor_token_ids), std::sqrt((float)c));
    auto * codes = round_state(ggml_add(ctx,
        ggml_mul(ctx, normalized_code, ggml_repeat(ctx, in.edge_mask, normalized_code)),
        ggml_mul(ctx, ggml_cast(ctx, anchor_code, GGML_TYPE_F32),
                 ggml_repeat(ctx, in.anchor_mask, normalized_code))));

    // Independent learned receiver, predecessor gate, and value.
    auto * receiver = round_state(ggml_mul_mat(ctx, w.gate_w, codes));
    auto * sender   = round_state(ggml_mul_mat(ctx, w.sender_w, codes));
    auto * value    = round_state(ggml_mul_mat(ctx, w.value_w, codes));
    auto * packed   = ggml_cont(ctx, ggml_concat(ctx,
            ggml_concat(ctx, receiver, sender, 0), value, 0));

    // Previous row selection is explicit. The first row of each block is
    // masked to zero, so it never observes the preceding block's state.
    auto * sender_prev = ggml_get_rows(ctx, ggml_cont(ctx, sender), in.predecessor_ids);
    auto * value_prev  = ggml_get_rows(ctx, ggml_cont(ctx, value),  in.predecessor_ids);
    auto * gate_f32 = ggml_sigmoid(ctx, ggml_add(ctx,
        ggml_add(ctx, ggml_cast(ctx, receiver, GGML_TYPE_F32),
                      ggml_cast(ctx, sender_prev, GGML_TYPE_F32)),
        ggml_repeat(ctx, ggml_cast(ctx, w.gate_bias, GGML_TYPE_F32), receiver)));
    auto * message = round_state(ggml_mul(ctx,
        ggml_mul(ctx, gate_f32, ggml_cast(ctx, value_prev, GGML_TYPE_F32)),
        ggml_repeat(ctx, in.edge_mask, gate_f32)));

    auto * delta = round_state(ggml_mul_mat(ctx, w.write_w, message));
    auto * updated = round_state(ggml_add(ctx, raw_f32,
        ggml_mul(ctx, ggml_cast(ctx, delta, GGML_TYPE_F32), ggml_repeat(ctx, rms, raw_f32))));

    // Tencent feature_finish: round residual to BF16, RMSNorm, then next
    // layer's normalization weight, with BF16 output.
    // CUDA's ggml_cuda_op_rms_norm() accepts F32 input/output only.
    // The learned residual is BF16 at this point; applying RMSNorm before
    // casting to F32 fatally asserts inside the Linux CUDA RPC worker and
    // closes the socket during GRAPH_COMPUTE. Tencent's feature_finish
    // likewise normalizes the BF16-rounded residual using F32 arithmetic.
    auto * next = ggml_rms_norm(ctx, ggml_cast(ctx, updated, GGML_TYPE_F32), norm_eps);
    // Multiply in F32 then round the completed next-layer activation to BF16.
    auto * normalized = round_state(ggml_mul(ctx,
        next, ggml_cast(ctx, w.next_norm_w, GGML_TYPE_F32)));

    return {raw, rms, packed, updated, normalized};
}

} // namespace llama_dspine
