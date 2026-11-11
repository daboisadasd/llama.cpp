// SPDX-License-Identifier: Apache-2.0
// Native GGML DSpine K-way serial refinement, CPU reference smoke.
// Uses tiny independent F32 synthetic weights; NOT the trained model.
#include "ggml.h"
#include "ggml-backend.h"
#include "dspine-refine.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
static ggml_tensor * input_2d(ggml_context * ctx, ggml_type type, int64_t x, int64_t y) {
    auto * t = ggml_new_tensor_2d(ctx, type, x, y);
    ggml_set_input(t);
    return t;
}
static ggml_tensor * input_1d(ggml_context * ctx, ggml_type type, int64_t n) {
    auto * t = ggml_new_tensor_1d(ctx, type, n);
    ggml_set_input(t);
    return t;
}
template<class T> static void fill(ggml_tensor * tensor, const std::vector<T> & value) {
    check(tensor->buffer != nullptr, "input buffer missing (did graph allocation run?)");
    check(ggml_nelements(tensor) == (int64_t)value.size(), "bad input shape");
    ggml_backend_tensor_set(tensor, value.data(), 0, value.size() * sizeof(T));
}

int main() {
    try {
        constexpr int64_t H=4, R=2, V=8, K=2, T=3, N=T;
        ggml_init_params ip = {};
        ip.mem_size = 4 * 1024 * 1024;
        ip.no_alloc = true;
        auto * ctx = ggml_init(ip);
        check(ctx != nullptr, "GGML context initialization failed");

        llama_dspine::refine_weights w;
        w.lm_head = input_2d(ctx, GGML_TYPE_F32, H, V);
        w.norm_w = input_1d(ctx, GGML_TYPE_F32, H);
        w.tg = input_2d(ctx, GGML_TYPE_F32, R, V);
        w.tsv = input_2d(ctx, GGML_TYPE_F32, 2*R, V);
        w.gate_bias = input_1d(ctx, GGML_TYPE_F32, R);

        llama_dspine::refine_inputs in;
        in.raw = input_2d(ctx, GGML_TYPE_F32, H, N);
        in.rms = input_2d(ctx, GGML_TYPE_F32, 1, N);
        in.packed = input_2d(ctx, GGML_TYPE_F32, 3*R, N);
        in.candidate_ids = input_2d(ctx, GGML_TYPE_I32, K, T-1);
        in.previous_ids = input_2d(ctx, GGML_TYPE_I32, K, T-1);

        auto * table = llama_dspine::build_serial_table(ctx, in, w, T, K);
        check(table->ne[0] == K && table->ne[1] == T-1, "bad transition table shape");
        ggml_set_output(table);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, table);
        ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        check(backend != nullptr, "CPU backend init failed");
        ggml_backend_sched_t sched = ggml_backend_sched_new(&backend, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
        check(sched != nullptr, "GGML scheduler initialization failed");
        // Never invoke side-effectful allocation/execution inside C assert():
        // MSVC Release would eliminate the call.
        check(ggml_backend_sched_alloc_graph(sched, graph), "GGML allocation failed");

        std::vector<float> head(H*V, 0.0f), norm(H, 1.0f);
        head[H*2 + 0] = 4.f;
        head[H*3 + 0] = 1.f;
        head[H*4 + 0] = 1.f;
        head[H*5 + 0] = 3.f;
        fill(w.lm_head, head);
        fill(w.norm_w, norm);

        std::vector<float> tg(R*V, 0.0f), tsv(2*R*V, 0.0f);
        tg[R*4 + 0] = 2.f;
        // token ID 3 is a previous-lane option at step 1. Its sender
        // value changes the next argmax from candidate 5 to candidate 4.
        tsv[(2*R)*3 + R] = 10.f;
        fill(w.tg, tg);
        fill(w.tsv, tsv);
        fill(w.gate_bias, std::vector<float>(R, 0));

        // raw rows: row 0 anchor; steps 1,2 both point in hidden dim0.
        fill(in.raw, std::vector<float>{
            1,0,0,0,
            1,0,0,0,
            1,0,0,0
        });
        fill(in.rms, std::vector<float>{1,1,1});
        fill(in.packed, std::vector<float>(3*R*N, 0));
        std::vector<int32_t> cand = {2,3, 4,5};
        fill(in.candidate_ids, cand);
        fill(in.previous_ids, std::vector<int32_t>{1,1, 2,3});

        check(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS,
              "refinement graph execution failed");
        std::vector<int32_t> index(K*(T-1));
        ggml_backend_tensor_get(table, index.data(), 0, index.size()*sizeof(int32_t));
        if (index != std::vector<int32_t>({0,0,1,0})) {
            std::fprintf(stderr, "Unexpected DSpine indices [%d,%d,%d,%d]\n",
                         index[0],index[1],index[2],index[3]);
            return 2;
        }
        auto chain = llama_dspine::serial_chain(index, cand, T, K);
        check(chain == std::vector<int32_t>({2,5}), "wrong DSpine greedy token chain");

        // A second block must start its transition lane independently.
        const auto two_blocks = llama_dspine::serial_chain(
            {0,0,1,0, 1,1,0,1},
            {2,3,4,5, 6,7,8,9}, T, K);
        check(two_blocks == std::vector<int32_t>({2,5, 7,9}),
              "DSpine blocks share a predecessor state (must not leak)");

        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_free(ctx);
        std::puts("DSpine K-way serial GGML scoring, chain walking and block boundaries: PASS");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "DSpine refinement test FAILED: %s\n", e.what());
        return 1;
    }
}
