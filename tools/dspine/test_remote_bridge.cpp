// SPDX-License-Identifier: Apache-2.0
// Run Tencent's *actual* learned DSpine last-write bridge on a Linux RPC GPU.
//
// This is a native GGML backend/RPC graph test, NOT a full DSpine drafter:
// no five-layer attention decoder, no serial refinement, no target verification.
// It intentionally uses GGUF's low-level API (general.architecture=dspine is
// not yet registered as a llama_model) and uploads only the selected weights.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "gguf.h"
#include "dspine-bridge.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using ctx_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

constexpr int64_t N = 16;
constexpr int64_t H = 5120;
constexpr int64_t C = 1024;
constexpr int64_t R = 512;
constexpr int64_t V = 248320;
constexpr size_t CHUNK = 4 * 1024 * 1024;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

ggml_tensor * require_weight(ggml_context * wctx, gguf_context * gguf, const std::string & name,
                             ggml_type type, const std::vector<int64_t> & dims) {
    auto * t = ggml_get_tensor(wctx, name.c_str());
    const int64_t id = gguf_find_tensor(gguf, name.c_str());
    require(id >= 0 && t != nullptr, "GGUF weight missing: " + name);
    require(t->type == type ||
            (type == GGML_TYPE_BF16 && t->type == GGML_TYPE_F32),
            "wrong tensor type for " + name + ": expected BF16 or F32");
    for (size_t i = 0; i < dims.size(); i++) {
        require(t->ne[i] == dims[i], "wrong shape for " + name);
    }
    for (size_t i = dims.size(); i < GGML_MAX_DIMS; i++) {
        require(t->ne[i] == 1, "unexpected extra tensor axis for " + name);
    }
    ggml_set_input(t);
    return t;
}

ggml_tensor * input_f32(ggml_context * ctx, int64_t dim0, int64_t dim1) {
    auto * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim0, dim1);
    ggml_set_input(t);
    return t;
}

ggml_tensor * input_i32(ggml_context * ctx, int64_t n) {
    auto * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_input(t);
    return t;
}

void upload_tensor(std::ifstream & stream, gguf_context * gguf, ggml_tensor * dst) {
    const int64_t index = gguf_find_tensor(gguf, dst->name);
    require(index >= 0, std::string("GGUF index missing for ") + dst->name);
    const size_t absolute = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, index);
    const size_t nbytes = ggml_nbytes(dst);
    require(nbytes == gguf_get_tensor_size(gguf, index),
            std::string("GGUF tensor size mismatch for ") + dst->name);
    require(dst->buffer != nullptr, std::string("RPC buffer absent for ") + dst->name);
    stream.clear();
    stream.seekg((std::streamoff)absolute, std::ios::beg);
    require(stream.good(), std::string("cannot seek to tensor ") + dst->name);
    std::vector<char> chunk(CHUNK);
    for (size_t offset = 0; offset < nbytes; ) {
        const size_t n = std::min(CHUNK, nbytes - offset);
        stream.read(chunk.data(), (std::streamsize)n);
        require(stream.gcount() == (std::streamsize)n,
                std::string("truncated GGUF tensor payload: ") + dst->name);
        ggml_backend_tensor_set(dst, chunk.data(), offset, n);
        offset += n;
    }
    std::printf("uploaded %-60s %.2f MiB\n", dst->name, (double)nbytes/(1024*1024));
}

template <typename T>
void put(ggml_tensor * t, const std::vector<T> & v) {
    require(t->buffer != nullptr, "input tensor buffer not allocated");
    require(ggml_nelements(t) == (int64_t)v.size(), "input element count mismatch");
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(T));
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
            "Usage: test-dspine-rpc-bridge.exe DSpine-b16.gguf WORKER_IP:50052\n"
            "Only tests the genuine final DSpine code-bridge graph, not DSpine inference.\n");
        return 2;
    }
    try {
        const std::string filename = argv[1], worker = argv[2];
        require(worker.find(':') != std::string::npos, "Expected private worker endpoint host:port");
        ggml_context * wctx_raw = nullptr;
        gguf_init_params args = {};
        args.no_alloc = true;
        args.ctx = &wctx_raw;
        gguf_ptr uf(gguf_init_from_file(filename.c_str(), args), gguf_free);
        require(uf != nullptr && wctx_raw != nullptr, "Could not parse DSpine GGUF metadata");
        ctx_ptr wctx(wctx_raw, ggml_free);
        const int64_t arch_id = gguf_find_key(uf.get(), "general.architecture");
        require(arch_id >= 0 && gguf_get_kv_type(uf.get(), arch_id) == GGUF_TYPE_STRING &&
                std::string(gguf_get_val_str(uf.get(), arch_id)) == "dspine",
                "Refusing non-DSpine or incomplete GGUF");
        const int64_t v_id = gguf_find_key(uf.get(), "dspine.format_version");
        require(v_id >= 0 && gguf_get_val_u32(uf.get(), v_id) == 1,
                "Unknown DSpine format version");

        // Last-write weights (layer index 4) avoid any ambiguity around
        // the earlier-layer read and next-norm offsets in this first test.
        const std::string prefix = "model.selected_prefix_bridge.";
        auto * codebook = require_weight(wctx.get(), uf.get(), prefix + "codebook",
                                        GGML_TYPE_BF16, {C,V});
        auto * read_w = require_weight(wctx.get(), uf.get(), prefix + "read_weight",
                                      GGML_TYPE_BF16, {H,C});
        auto * gate_w = require_weight(wctx.get(), uf.get(), prefix + "gate.weight",
                                      GGML_TYPE_BF16, {C,R});
        auto * sender_w = require_weight(wctx.get(), uf.get(), prefix + "predecessor_gate.weight",
                                        GGML_TYPE_BF16, {C,R});
        auto * value_w = require_weight(wctx.get(), uf.get(), prefix + "prev.weight",
                                       GGML_TYPE_BF16, {C,R});
        auto * bias = require_weight(wctx.get(), uf.get(), prefix + "gate.bias",
                                     GGML_TYPE_BF16, {R});
        auto * write_all = require_weight(wctx.get(), uf.get(), prefix + "feature_write_weights",
                                          GGML_TYPE_BF16, {R,H,5});
        auto * norm = require_weight(wctx.get(), uf.get(), "model.norm.weight",
                                     GGML_TYPE_BF16, {H});

        ggml_backend_load_all();
        ggml_backend_reg_t rpc = ggml_backend_rpc_add_server(worker.c_str());
        require(rpc != nullptr, "Failed to connect/register GGML RPC worker");
        ggml_backend_register(rpc);
        auto * device = ggml_backend_dev_by_name("RPC0");
        require(device != nullptr, "No RPC0 device registered");
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_dev_memory(device, &free_mem, &total_mem);
        std::printf("Remote %s GPU: %.1f GiB free / %.1f GiB total\n", worker.c_str(),
                    (double)free_mem/(1024*1024*1024), (double)total_mem/(1024*1024*1024));
        std::unique_ptr<std::remove_pointer_t<ggml_backend_t>, decltype(&ggml_backend_free)>
            backend(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
        require(backend != nullptr, "GGML RPC backend init failed");

        ggml_init_params gargs = {};
        gargs.mem_size = 64 * 1024 * 1024;
        gargs.no_alloc = true;
        ctx_ptr graph_ctx(ggml_init(gargs), ggml_free);
        require(graph_ctx != nullptr, "GGML graph context initialization failed");
        auto * ctx = graph_ctx.get();

        auto * last_write = ggml_view_2d(ctx, write_all, R, H, write_all->nb[1],
                                         (size_t)4 * write_all->nb[2]);
        ggml_set_name(last_write, "last_code_write");

        llama_dspine::bridge_weights w;
        w.codebook = codebook;
        w.read_w = read_w;
        w.gate_w = gate_w;
        w.sender_w = sender_w;
        w.value_w = value_w;
        w.gate_bias = bias;
        w.write_w = last_write;
        w.next_norm_w = norm;

        llama_dspine::bridge_inputs in;
        in.hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, H, N);
        ggml_set_input(in.hidden);
        in.anchor_token_ids = input_i32(ctx, N);
        in.predecessor_ids = input_i32(ctx, N);
        in.anchor_mask = input_f32(ctx, 1, N);
        in.edge_mask = input_f32(ctx, 1, N);
        in.eps = input_f32(ctx, 1, 1);
        // Full config.eps is also carried in GGUF as config_json.
        // The official Qwen3.8 DSpine config uses 1e-6. A future decoder
        // must derive epsilon from the validated config, not assume it.
        auto output = llama_dspine::build_bridge(ctx, in, w, 16, 1e-6f);
        ggml_set_output(output.residual);
        ggml_set_output(output.normalized);
        ggml_set_output(output.packed);
        ggml_set_output(output.rms);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output.residual);
        ggml_build_forward_expand(graph, output.normalized);
        ggml_build_forward_expand(graph, output.packed);
        ggml_build_forward_expand(graph, output.rms);

        ggml_backend_t raw_backend = backend.get();
        auto sched = ggml_backend_sched_new(&raw_backend, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
        require(sched != nullptr, "GGML scheduler init failed");
        std::unique_ptr<std::remove_pointer_t<ggml_backend_sched_t>, decltype(&ggml_backend_sched_free)>
            scheduler(sched, ggml_backend_sched_free);
        require(ggml_backend_sched_alloc_graph(scheduler.get(), graph),
                "Failed to allocate RPC graph and weights");

        std::ifstream file(filename, std::ios::binary);
        require(file.good(), "Cannot open GGUF for weight streaming");
        for (ggml_tensor * weight : {codebook, read_w, gate_w, sender_w, value_w,
                                      bias, write_all, norm}) {
            upload_tensor(file, uf.get(), weight);
        }

        std::vector<ggml_bf16_t> hidden((size_t)(H*N));
        for (int64_t row=0; row<N; row++) for(int64_t col=0; col<H; col++) {
            const float x = 0.125f * (float)((col % 11) - 5) +
                            0.0125f*(float)(row+1);
            hidden[(size_t)(row*H+col)] = ggml_fp32_to_bf16(x);
        }
        put(in.hidden, hidden);
        std::vector<int32_t> anchors((size_t)N, 0), predecessors((size_t)N, 0);
        std::vector<float> anchor_mask((size_t)N, 0), edge_mask((size_t)N, 1);
        for (int i = 0; i < N; i++) {
            predecessors[(size_t)i] = i == 0 ? 0 : i-1;
        }
        anchor_mask[0] = 1;
        edge_mask[0] = 0;
        put(in.anchor_token_ids, anchors);
        put(in.predecessor_ids, predecessors);
        put(in.anchor_mask, anchor_mask);
        put(in.edge_mask, edge_mask);
        put(in.eps, std::vector<float>{1e-6f});

        require(ggml_backend_sched_graph_compute(scheduler.get(), graph) == GGML_STATUS_SUCCESS,
                "Remote bridge graph execution failed");
        std::vector<ggml_bf16_t> residual((size_t)(H*N));
        std::vector<float> rms((size_t)N);
        ggml_backend_tensor_get(output.residual, residual.data(), 0, residual.size()*sizeof(ggml_bf16_t));
        ggml_backend_tensor_get(output.rms, rms.data(), 0, rms.size()*sizeof(float));
        float checksum = 0;
        for (auto x : residual) {
            const float y = ggml_bf16_to_fp32(x);
            require(std::isfinite(y), "non-finite remote DSpine bridge output");
            checksum += y;
        }
        for (auto v : rms) require(std::isfinite(v), "non-finite bridge RMS");
        std::printf("DSpine last-layer learned GGML bridge on RPC0: PASS\n");
        std::printf("  first residual element: %.6f; first RMS: %.6f; checksum: %.6f\n",
                    ggml_bf16_to_fp32(residual.front()), rms.front(), checksum);
        std::puts("  NOTE: this tests real bridge weights and remote GPU graph execution,"
                  " NOT full DSpine drafting or verification.");
        return 0;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "DSpine RPC bridge test FAILED: %s\n", exc.what());
        return 1;
    }
}
