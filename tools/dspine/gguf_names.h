// SPDX-License-Identifier: Apache-2.0
#pragma once

// Tencent's safetensors exports may either keep or strip the leading
// "model." namespace. The native DSpine GGUF converter faithfully preserves
// the checkpoint's original tensor names, rather than silently rewriting
// them, so the standalone GGUF/RPC runner must handle both forms.
//
// This resolver refuses ambiguous files containing BOTH variants. It does
// not guess unrelated names or permit loading the wrong tensor.
#include "gguf.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace llama_dspine {

struct resolved_tensor_name {
    std::string name;
    int64_t index = -1;
};

static inline resolved_tensor_name resolve_gguf_tensor_name(
        const gguf_context * uf, const std::string & requested) {
    if (!uf || requested.empty()) {
        throw std::invalid_argument("DSpine GGUF tensor request missing context or name");
    }
    const std::string prefix = "model.";
    const std::string alternate = requested.compare(0, prefix.size(), prefix) == 0
        ? requested.substr(prefix.size()) : prefix + requested;
    const int64_t exact = gguf_find_tensor(uf, requested.c_str());
    const int64_t other = gguf_find_tensor(uf, alternate.c_str());
    if (exact >= 0 && other >= 0) {
        throw std::runtime_error("Ambiguous DSpine GGUF tensor: both '" +
                                 requested + "' and '" + alternate + "' exist");
    }
    if (exact >= 0) return {requested, exact};
    if (other >= 0) return {alternate, other};
    throw std::runtime_error("DSpine GGUF weight missing: '" + requested +
                             "' (also checked '" + alternate + "')");
}

} // namespace llama_dspine
