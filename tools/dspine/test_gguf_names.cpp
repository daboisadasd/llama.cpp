// SPDX-License-Identifier: Apache-2.0
// Fail-closed regression tests for the original Tencent DSpine safetensors
// namespace when preserved inside our native GGUF converter.
#include "ggml.h"
#include "gguf.h"
#include "gguf_names.h"

#include <cstdio>
#include <stdexcept>
#include <string>

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
static void add(ggml_context * ctx, gguf_context * uf, const char * name) {
    auto * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(t, name);
    gguf_add_tensor(uf, t);
}
static void resolves(gguf_context * uf, const std::string & requested, const char * expected) {
    const auto v = llama_dspine::resolve_gguf_tensor_name(uf, requested);
    check(v.name == expected, "wrong resolved tensor name");
    check(v.index == gguf_find_tensor(uf, expected), "wrong resolved tensor index");
}
int main() {
    ggml_init_params params = {};
    params.mem_size = 1024*1024;
    params.no_alloc = true;
    auto * ctx = ggml_init(params);
    if (!ctx) return 1;
    auto * uf = gguf_init_empty();
    if (!uf) return 2;
    try {
        // The downloaded Tencent GGUF in this project contains names
        // such as selected_prefix_bridge.codebook, NOT model.selected_...
        add(ctx, uf, "selected_prefix_bridge.codebook");
        add(ctx, uf, "selected_prefix_bridge.independent_write_version");
        add(ctx, uf, "selected_prefix_bridge.out.weight");
        add(ctx, uf, "norm.weight");
        add(ctx, uf, "output.weight");
        resolves(uf, "model.selected_prefix_bridge.codebook",
                 "selected_prefix_bridge.codebook");
        resolves(uf, "model.selected_prefix_bridge.independent_write_version",
                 "selected_prefix_bridge.independent_write_version");
        resolves(uf, "model.selected_prefix_bridge.out.weight",
                 "selected_prefix_bridge.out.weight");
        resolves(uf, "model.norm.weight", "norm.weight");
        resolves(uf, "output.weight", "output.weight");

        // Equally support checkpoints that preserve an explicit model. prefix.
        add(ctx, uf, "model.layers.0.self_attn.q_proj.weight");
        resolves(uf, "layers.0.self_attn.q_proj.weight",
                 "model.layers.0.self_attn.q_proj.weight");

        bool missing_rejected = false;
        try {
            (void)llama_dspine::resolve_gguf_tensor_name(uf,
                       "model.selected_prefix_bridge.missing_weight");
        } catch (const std::runtime_error &) {
            missing_rejected = true;
        }
        check(missing_rejected, "missing weight must fail closed");
        // Two names that become identical after prefix normalization are
        // ambiguous: reject rather than silently using either version.
        add(ctx, uf, "model.selected_prefix_bridge.codebook");
        bool ambiguous_rejected = false;
        try {
            (void)llama_dspine::resolve_gguf_tensor_name(uf,
                       "model.selected_prefix_bridge.codebook");
        } catch (const std::runtime_error &) {
            ambiguous_rejected = true;
        }
        check(ambiguous_rejected, "ambiguous weight must fail closed");
        gguf_free(uf);
        ggml_free(ctx);
        std::puts("DSpine GGUF tensor namespace compatibility: PASS");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "DSpine GGUF name resolution FAILED: %s\n", e.what());
        gguf_free(uf);
        ggml_free(ctx);
        return 3;
    }
}
