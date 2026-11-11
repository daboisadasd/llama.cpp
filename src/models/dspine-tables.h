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
#include <algorithm>
#include <cmath>
#include <vector>

namespace llama_dspine {

struct table_result {
    ggml_tensor * tg = nullptr; // [R,V] BF16
    ggml_tensor * tsv = nullptr; // [2R,V] BF16
};

// Concatenate vocabulary tiles in a balanced tree to bound the lifetime and
// copying cost of intermediate concatenations. All tiles share ne[0] and
// are kept in original token-id order.
static inline ggml_tensor * concat_vocab_tiles(
        ggml_context * ctx, std::vector<ggml_tensor *> tiles) {
    GGML_ASSERT(ctx && !tiles.empty());
    while (tiles.size() > 1) {
        std::vector<ggml_tensor *> next;
        next.reserve((tiles.size() + 1)/2);
        for (size_t i = 0; i < tiles.size(); i += 2) {
            next.push_back(i+1 < tiles.size()
                ? ggml_concat(ctx, tiles[i], tiles[i+1], 1)
                : tiles[i]);
        }
        tiles.swap(next);
    }
    return tiles.front();
}

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

    // The remote 3080 Ti has ~12 GiB VRAM, but the target BF16 head
    // alone occupies ~2.4 GiB. Casting the ENTIRE head to F32 and
    // materializing its normalizer would require multiple 4.7 GiB buffers
    // and crash the remote worker during -Tail. Tencent's reference uses
    // vocabulary chunks for this reason. Keep intermediate math inside
    // 8192-row tiles, retaining the exact F32 -> BF16 rounding boundary.
    constexpr int64_t TILE = 8192;
    auto * norm_f32 = ggml_cast(ctx, norm_w, GGML_TYPE_F32);
    auto * write_t = ggml_cont(ctx, ggml_transpose(ctx, write_w));
    std::vector<ggml_tensor *> tg_tiles;
    std::vector<ggml_tensor *> tsv_tiles;
    const int64_t n_tiles = (v + TILE - 1) / TILE;
    tg_tiles.reserve((size_t)n_tiles);
    tsv_tiles.reserve((size_t)n_tiles);

    for (int64_t beg = 0; beg < v; beg += TILE) {
        const int64_t count = std::min(TILE, v - beg);

        auto * head_tile = ggml_view_2d(ctx, target_head, h, count,
                target_head->nb[1], (size_t)beg * target_head->nb[1]);
        auto * head_f32 = ggml_cast(ctx, head_tile, GGML_TYPE_F32);
        // ggml_mul broadcasts the [H] normalizer across vocabulary rows,
        // avoiding an enormous explicit GGML_OP_REPEAT buffer.
        auto * weighted_head = ggml_cast(ctx,
            ggml_mul(ctx, head_f32, norm_f32), GGML_TYPE_BF16);
        tg_tiles.push_back(ggml_cast(ctx,
            ggml_mul_mat(ctx, write_t, weighted_head), GGML_TYPE_BF16));

        auto * code_tile = ggml_view_2d(ctx, codebook, c, count,
                codebook->nb[1], (size_t)beg * codebook->nb[1]);
        auto * scaled_codes = ggml_cast(ctx,
            ggml_scale(ctx, ggml_cast(ctx, code_tile, GGML_TYPE_F32),
                       std::sqrt((float)c)), GGML_TYPE_BF16);
        auto * gate = ggml_mul_mat(ctx, sender_w, scaled_codes);
        auto * value = ggml_mul_mat(ctx, value_w, scaled_codes);
        tsv_tiles.push_back(ggml_concat(ctx,
            ggml_cast(ctx, gate, GGML_TYPE_BF16),
            ggml_cast(ctx, value, GGML_TYPE_BF16), 0));
    }

    auto * tg = concat_vocab_tiles(ctx, std::move(tg_tiles));
    auto * tsv = concat_vocab_tiles(ctx, std::move(tsv_tiles));
    GGML_ASSERT(tg->ne[0] == r && tg->ne[1] == v);
    GGML_ASSERT(tsv->ne[0] == 2*r && tsv->ne[1] == v);
    return {tg, tsv};
}

} // namespace llama_dspine
