// SPDX-License-Identifier: Apache-2.0
// Execute the first DSpine learned-code GGML graph on the CPU, using tiny
// synthetic tensors. This is NOT a complete 5-layer DSpine model.
#include "ggml.h"
#include "ggml-backend.h"
#include "dspine-bridge.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void set_floats(ggml_tensor * tensor, const std::vector<float> & data) {
    assert(ggml_nelements(tensor) == (int64_t)data.size());
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
}
static void set_ints(ggml_tensor * tensor, const std::vector<int32_t> & data) {
    assert(ggml_nelements(tensor) == (int64_t)data.size());
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(int32_t));
}
static ggml_tensor * x1(ggml_context * ctx, enum ggml_type type, int64_t n) {
    auto * tensor = ggml_new_tensor_1d(ctx, type, n);
    ggml_set_input(tensor);
    return tensor;
}
static ggml_tensor * x2(ggml_context * ctx, enum ggml_type type, int64_t x, int64_t y) {
    auto * tensor = ggml_new_tensor_2d(ctx, type, x, y);
    ggml_set_input(tensor);
    return tensor;
}

int main() {
    // CPU/F32 identity contract: all bridge weights zero means the residual is
    // unchanged, and the returned next-layer state is its RMS normalization.
    ggml_init_params params = {};
    params.mem_size = 32 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    assert(ctx);
    constexpr int64_t H = 4, C = 4, R = 2, V = 8, T = 3, N = 2 * T;

    llama_dspine::bridge_inputs in;
    in.hidden = x2(ctx, GGML_TYPE_F32, H, N);
    in.anchor_token_ids = x1(ctx, GGML_TYPE_I32, N);
    in.predecessor_ids = x1(ctx, GGML_TYPE_I32, N);
    in.anchor_mask = x2(ctx, GGML_TYPE_F32, 1, N);
    in.edge_mask = x2(ctx, GGML_TYPE_F32, 1, N);
    in.eps = x1(ctx, GGML_TYPE_F32, 1);
    llama_dspine::bridge_weights weights;
    weights.codebook = x2(ctx, GGML_TYPE_F32, C, V);
    weights.read_w = x2(ctx, GGML_TYPE_F32, H, C);
    weights.gate_w = x2(ctx, GGML_TYPE_F32, C, R);
    weights.sender_w = x2(ctx, GGML_TYPE_F32, C, R);
    weights.value_w = x2(ctx, GGML_TYPE_F32, C, R);
    weights.gate_bias = x1(ctx, GGML_TYPE_F32, R);
    weights.write_w = x2(ctx, GGML_TYPE_F32, R, H);
    weights.next_norm_w = x1(ctx, GGML_TYPE_F32, H);

    auto out = llama_dspine::build_bridge(ctx, in, weights, (int)T, 1e-6f);
    assert(out.raw->ne[0] == H && out.raw->ne[1] == N);
    assert(out.rms->ne[0] == 1 && out.rms->ne[1] == N);
    assert(out.packed->ne[0] == 3*R && out.packed->ne[1] == N);
    assert(out.residual->ne[0] == H && out.residual->ne[1] == N);
    assert(out.normalized->ne[0] == H && out.normalized->ne[1] == N);

    ggml_set_output(out.residual);
    ggml_set_output(out.normalized);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out.residual);
    ggml_build_forward_expand(graph, out.normalized);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(backend);
    ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
    assert(sched);
    assert(ggml_backend_sched_alloc_graph(sched, graph));

    const std::vector<float> hidden = {
        1.f,2.f,3.f,4.f,
        1.f,1.f,1.f,1.f,
        2.f,2.f,2.f,2.f,
        1.f,1.f,0.f,0.f,
        5.f,0.f,0.f,0.f,
        1.f,1.f,1.f,1.f,
    };
    set_floats(in.hidden, hidden);
    set_ints(in.anchor_token_ids, {1,0,0,2,0,0});
    set_ints(in.predecessor_ids, {0,0,1,3,3,4});
    set_floats(in.anchor_mask, {1,0,0,1,0,0});
    set_floats(in.edge_mask, {0,1,1,0,1,1});
    set_floats(in.eps, {1e-6f});

    auto zeros = [&](ggml_tensor * tensor) {
        set_floats(tensor, std::vector<float>((size_t)ggml_nelements(tensor), 0.0f));
    };
    zeros(weights.codebook);
    zeros(weights.read_w);
    zeros(weights.gate_w);
    zeros(weights.sender_w);
    zeros(weights.value_w);
    zeros(weights.gate_bias);
    zeros(weights.write_w);
    set_floats(weights.next_norm_w, {1.f,1.f,1.f,1.f});

    assert(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> actual(hidden.size());
    std::vector<float> normalized(hidden.size());
    ggml_backend_tensor_get(out.residual, actual.data(), 0, actual.size()*sizeof(float));
    ggml_backend_tensor_get(out.normalized, normalized.data(), 0, normalized.size()*sizeof(float));
    for (size_t i = 0; i < hidden.size(); ++i) {
        if (std::abs(actual[i] - hidden[i]) > 1e-4f) {
            std::fprintf(stderr, "DSpine bridge identity violated at %zu (%f vs %f)\n", i, actual[i], hidden[i]);
            return 2;
        }
    }
    for (int64_t row = 0; row < N; ++row) {
        float s = 0;
        for (int64_t i = 0; i < H; ++i) {
            const float x = hidden[(size_t)(row*H+i)];
            s += x*x;
        }
        const float inv = 1.f/std::sqrt(s/H + 1e-6f);
        for (int64_t i = 0; i < H; ++i) {
            const size_t j = (size_t)(row*H+i);
            if (std::abs(normalized[j] - hidden[j]*inv) > 1e-3f) {
                std::fprintf(stderr, "DSpine bridge normalization mismatch at row %lld item %lld\n",
                             (long long)row, (long long)i);
                return 3;
            }
        }
    }
    // Second pass: turn on just one sender and one learned write. In block 0,
    // anchor token 1 supplies code[0]=sqrt(C)=2. With gate=0, sigmoid=0.5,
    // message(row 1)[0]=1 and delta[0]=2. Because RMS(row 1)=1, hidden
    // row 1 becomes [3,1,1,1]. Block 1 uses a different anchor token (2)
    // and must not inherit the previous block's final sender.
    std::vector<float> codebook((size_t)(C*V), 0.f);
    std::vector<float> value_w((size_t)(C*R), 0.f);
    std::vector<float> write_w((size_t)(R*H), 0.f);
    codebook[C] = 1.f;   // token id 1, first code dimension
    value_w[0] = 1.f;    // value projection: first code -> first rank
    write_w[0] = 2.f;    // learned write: first rank -> first hidden dimension
    set_floats(weights.codebook, codebook);
    set_floats(weights.value_w, value_w);
    set_floats(weights.write_w, write_w);
    if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "DSpine bridge nonzero write: compute failed\n");
        return 4;
    }
    ggml_backend_tensor_get(out.residual, actual.data(), 0, actual.size()*sizeof(float));
    for (int64_t row = 0; row < N; ++row) {
        for (int64_t i = 0; i < H; ++i) {
            const size_t j = (size_t)(row*H+i);
            const float expected = (row == 1 && i == 0) ? 3.f : hidden[j];
            if (std::abs(actual[j] - expected) > 1e-3f) {
                std::fprintf(stderr,
                             "DSpine learned adjacent write/boundary mismatch row=%lld dim=%lld (%f vs %f)\n",
                             (long long)row, (long long)i, actual[j], expected);
                return 5;
            }
        }
    }

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(ctx);
    std::puts("DSpine native GGML bridge CPU identity and normalization: PASS");
    return 0;
}
