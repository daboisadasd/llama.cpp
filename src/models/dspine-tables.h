// SPDX-License-Identifier: Apache-2.0
#pragma once

// Build Tencent DSpine's immutable serial refinement lookup tables in GGML.
// Evaluate ONCE per checkpoint/head combination and persist the two BF16
// results on the drafting device before starting generation. This helper
// only builds the graph: persistence, model loading, and caching belong to
// the native DSpine runner. Both tables must use the *same* BF16 output
// head as top-K and verification (not the Q4_K_M target storage).
//
// Tencent reference: sglang_dspine/kernels/dspine.py::build_serial_tables
// TG[v]  = BF16(BF16(lm_head[v].float * norm.float) @ write_bf16)
// TSV[v] = BF16(BF16(sqrt(C) * codebook[v].float) @ concat(sender, value)^T)
//
// This implements the tensor algebra in standard GGML nodes. Exact BF16
// rounding and CUDA matmul order still need comparison with Tencent's code.

#include "ggml.h"
#include <cmath>

namespace llama_dspine {

struct table_result {
    ggml_tensor * tg = nullptr; // [R,V] BF16
    ggml_tensor * tsv = nullptr; // [2R,V] BF16
};

static inline table_result build_serial_tables(
        ggml_context * ctx,
        ggml_tensor * target_head, // [H,V] BF16
        ggml_tensor * norm_w,      // [H] BF16
        ggml_tensor * write_w,     // [R,H] BF16, last code write
        ggml_tensor * codebook,    // [C,V] BF16
        ggml_tensor * sender_w,    // [C,R] BF16
        ggml_tensor * value_w) {  // [C,R] BF16
    GGML_ASSERT(ctx && target_head && norm_w && write_w && codebook && sender_w && value_w);
    const int64_t h = target_head->ne[0];
    const int64_t v = target_head->ne[1];
    const int64_t r = write_w->ne[0];
    const int64_t c = codebook->ne[0];
    GGML_ASSERT(v > 0 && r > 0 && c > 0);
    GGML_ASSERT(target_head->type == GGML_TYPE_BF16 && codebook->type == GGML_TYPE_BF16);
    GGML_ASSERT(norm_w->ne[0] == h);
    GGML_ASSERT(write_w->ne[1] == h);
    GGML_ASSERT(codebook->ne[1] == v);
    GGML_ASSERT(sender_w->ne[0] == c && sender_w->ne[1] == r);
    GGML_ASSERT(value_w->ne[0] == c && value_w->ne[1] == r);

    // First emulate torch float multiplication followed by a BF16 cast.
    auto * norm_bf16 = ggml_cast(ctx, norm_w, GGML_TYPE_BF16);
    auto * weighted_head = ggml_mul(ctx, target_head, ggml_repeat(ctx, norm_bf16, target_head));
    weighted_head = ggml_cast(ctx, weighted_head, GGML_TYPE_BF16);

    // The GGML weight layout [R,H] is the transpose of Torch's [H,R].
    auto * write_t = ggml_cont(ctx, ggml_transpose(ctx, write_w));
    auto * tg = ggml_mul_mat(ctx, write_t, weighted_head);
    tg = ggml_cast(ctx, tg, GGML_TYPE_BF16);

    auto * scaled_codes = ggml_cast(ctx,
        ggml_scale(ctx, ggml_cast(ctx, codebook, GGML_TYPE_F32),
                   std::sqrt((float)c)), GGML_TYPE_BF16);
    auto * gate = ggml_mul_mat(ctx, sender_w, scaled_codes);
    auto * value = ggml_mul_mat(ctx, value_w, scaled_codes);
    auto * tsv = ggml_concat(ctx,
        ggml_cast(ctx, gate, GGML_TYPE_BF16),
        ggml_cast(ctx, value, GGML_TYPE_BF16), 0);
    GGML_ASSERT(tg->ne[0] == r && tg->ne[1] == v);
    GGML_ASSERT(tsv->ne[0] == 2*r && tsv->ne[1] == v);
    return {tg, tsv};
}

} // namespace llama_dspine
