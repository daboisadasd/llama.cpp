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
#include "dspine-tail.h"
#include "dspine-tables.h"
#include "gguf_names.h"

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
    // The DSpine converter preserves original safetensors names. Official
    // weights omit "model." on their code-bridge and model.norm tensors.
    const auto resolved = llama_dspine::resolve_gguf_tensor_name(gguf, name);
    auto * t = ggml_get_tensor(wctx, resolved.name.c_str());
    require(t != nullptr, "GGML tensor metadata missing: " + resolved.name);
    std::printf("weight %s -> %s\n", name.c_str(), resolved.name.c_str());
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

void validate_dspine_sentinel(std::ifstream & file, gguf_context * uf,
                              const std::string & name) {
    const auto resolved = llama_dspine::resolve_gguf_tensor_name(uf, name);
    const int64_t id = resolved.index;
    require(gguf_get_tensor_type(uf, id) == GGML_TYPE_I64 &&
            gguf_get_tensor_size(uf, id) == sizeof(int64_t),
            "DSpine version sentinel has unexpected type/size: " + name);
    const size_t offset = gguf_get_data_offset(uf) + gguf_get_tensor_offset(uf, id);
    file.clear();
    file.seekg((std::streamoff)offset, std::ios::beg);
    int64_t value = 0;
    file.read(reinterpret_cast<char *>(&value), sizeof(value));
    require(file.good() && value == 1, "Unsupported DSpine version sentinel: " + name);
}

void validate_terminal_disabled(std::ifstream & file, gguf_context * uf,
                                const std::string & name) {
    const auto resolved = llama_dspine::resolve_gguf_tensor_name(uf, name);
    const int64_t id = resolved.index;
    const ggml_type type = gguf_get_tensor_type(uf, id);
    require(type == GGML_TYPE_BF16 || type == GGML_TYPE_F32,
            "Unexpected DSpine terminal projection precision");
    const size_t width = type == GGML_TYPE_BF16 ? 2 : 4;
    const size_t nbytes = gguf_get_tensor_size(uf, id);
    require(nbytes > 0 && nbytes % width == 0, "Invalid terminal projection size");
    const size_t offset = gguf_get_data_offset(uf) + gguf_get_tensor_offset(uf, id);
    file.clear();
    file.seekg((std::streamoff)offset, std::ios::beg);
    std::vector<unsigned char> buf(1024*1024);
    for (size_t total=0; total<nbytes;) {
        const size_t n=std::min(buf.size(),nbytes-total);
        file.read(reinterpret_cast<char *>(buf.data()),(std::streamsize)n);
        require(file.gcount()==(std::streamsize)n, "Truncated terminal projection");
        for (size_t i=0; i<n; i+=width) {
            uint32_t bits=0;
            for (size_t j=0; j<width; j++) bits |= (uint32_t)buf[i+j] << (8*j);
            const uint32_t magnitude = type == GGML_TYPE_BF16 ? (bits & 0x7fff) : (bits & 0x7fffffff);
            require(magnitude == 0,
                    "DSpine terminal-disabled checkpoint has nonzero out.weight");
        }
        total+=n;
    }
}

template <typename T>
void put(ggml_tensor * t, const std::vector<T> & v) {
    require(t->buffer != nullptr, "input tensor buffer not allocated");
    require(ggml_nelements(t) == (int64_t)v.size(), "input element count mismatch");
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(T));
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr,
            "Usage: test-dspine-rpc-bridge.exe DSpine-b16.gguf WORKER_IP:50052 [--tail]\n"
            "Default: learned bridge; --tail: bridge, target top-K, TG/TSV and serial chain.\n"
            "No mode runs the full five-layer drafter or target verification yet.\n");
        return 2;
    }
    try {
        const std::string filename = argv[1], worker = argv[2];
        const bool tail_mode = argc == 4 && std::string(argv[3]) == "--tail";
        require(argc == 3 || tail_mode, "Unrecognized optional argument (expected --tail)");
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
        ggml_tensor * head = nullptr;
        if (tail_mode) {
            head = require_weight(wctx.get(), uf.get(), "output.weight",
                                  GGML_TYPE_BF16, {H,V});
            require(head->type == GGML_TYPE_BF16,
                    "DSpine serial refinement needs BF16, not quantized or F32 output.weight");
        }

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
        llama_dspine::bridge_result output;
        llama_dspine::tail_result tail;
        if (tail_mode) {
            // Pure GGML BF16 table construction + exact-sender top-K tail.
            // This is a single one-shot graph test; the production runtime
            // will precompute and cache TG/TSV only once during model load.
            auto tables = llama_dspine::build_serial_tables(
                ctx, head, norm, last_write, codebook, sender_w, value_w);
            llama_dspine::refine_weights rw = {
                head, norm, tables.tg, tables.tsv, bias
            };
            tail = llama_dspine::build_greedy_tail(ctx, in, w, rw, 16, 16, 1e-6f);
            output = tail.bridge;
            ggml_set_output(tail.candidate_ids);
            ggml_set_output(tail.transitions);
        } else {
            output = llama_dspine::build_bridge(ctx, in, w, 16, 1e-6f);
        }
        ggml_set_output(output.residual);
        ggml_set_output(output.normalized);
        ggml_set_output(output.packed);
        ggml_set_output(output.rms);

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output.residual);
        ggml_build_forward_expand(graph, output.normalized);
        ggml_build_forward_expand(graph, output.packed);
        ggml_build_forward_expand(graph, output.rms);
        if (tail_mode) {
            ggml_build_forward_expand(graph, tail.candidate_ids);
            ggml_build_forward_expand(graph, tail.transitions);
        }

        ggml_backend_t raw_backend = backend.get();
        auto sched = ggml_backend_sched_new(&raw_backend, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true);
        require(sched != nullptr, "GGML scheduler init failed");
        std::unique_ptr<std::remove_pointer_t<ggml_backend_sched_t>, decltype(&ggml_backend_sched_free)>
            scheduler(sched, ggml_backend_sched_free);
        require(ggml_backend_sched_alloc_graph(scheduler.get(), graph),
                "Failed to allocate RPC graph and weights");

        std::ifstream file(filename, std::ios::binary);
        require(file.good(), "Cannot open GGUF for weight streaming");
        validate_dspine_sentinel(file, uf.get(), prefix + "independent_write_version");
        validate_dspine_sentinel(file, uf.get(), prefix + "terminal_disabled_version");
        validate_terminal_disabled(file, uf.get(), prefix + "out.weight");
        for (ggml_tensor * weight : {codebook, read_w, gate_w, sender_w, value_w,
                                      bias, write_all, norm}) {
            upload_tensor(file, uf.get(), weight);
        }
        if (tail_mode) {
            upload_tensor(file, uf.get(), head);
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
        if (tail_mode) {
            const size_t n = (size_t)(16 * (N - 1));
            std::vector<int32_t> candidates(n), transitions(n);
            ggml_backend_tensor_get(tail.candidate_ids, candidates.data(),
                                    0, n * sizeof(int32_t));
            ggml_backend_tensor_get(tail.transitions, transitions.data(),
                                    0, n * sizeof(int32_t));
            for (int32_t token : candidates) {
                require(token >= 0 && token < V, "DSpine top-K produced out-of-vocabulary candidate");
            }
            const auto tokens = llama_dspine::serial_chain(transitions, candidates, 16, 16);
            require(tokens.size() == 15, "DSpine serial tail returned incorrect token count");
            std::printf("DSpine trained GGML bridge + 16-way greedy serial tail on RPC0: PASS\n");
            std::printf("  15 synthetic-input draft token IDs:");
            for (int32_t token : tokens) std::printf(" %d", token);
            std::printf("\n");
            std::puts("  NOTE: synthetic hidden input, not real 5-layer DSpine inference or target verification.");
        } else {
            std::puts("  NOTE: this tests real bridge weights and remote GPU graph execution,"
                      " NOT full DSpine drafting or verification.");
        }
        return 0;
    } catch (const std::exception & exc) {
        std::fprintf(stderr, "DSpine RPC bridge test FAILED: %s\n", exc.what());
        return 1;
    }
}
