#!/usr/bin/env python3
"""Audit the unmodified llama.cpp MTP-specific draft model/device code path."""
from pathlib import Path
import argparse
import json
import sys


def audit(root):
    root = Path(root)
    paths = {
        'args': root / 'common/arg.cpp',
        'spec': root / 'common/speculative.cpp',
        'context': root / 'src/llama-context.cpp',
        'server': root / 'tools/server/server-context.cpp',
    }
    texts = {k: p.read_text(encoding='utf-8') for k, p in paths.items()}
    rules = {
        'draft device CLI parsed': '--spec-draft-device' in texts['args'] and 'params.speculative.draft.devices = parse_device_list(value)' in texts['args'],
        'independent draft devices propagated': 'result.devices           = params_spec.devices;' in texts['spec'] and 'if (!params_spec.devices.empty())' in texts['spec'],
        'independent draft GGUF load': 'common_model_load(params.speculative.draft.mparams.get_paths(), mparams)' in texts['spec'],
        'MTP draft context on independent GGUF': 'llama_init_from_model(model_dft, cparams)' in texts['spec'],
        'MTP default reuses target weights': 'llama_init_from_model(model_tgt, cparams)' in texts['spec'],
        'MTP state bridge has host fallback': 'batch.set_embd_state(idx, { h_row, 1, (size_t) n_embd })' in texts['spec'],
        'backend state-src rejects unsupported device buffers': 'ggml_backend_supports_buft(backend.get(), buft)' in texts['context'],
        'MTP draft accepts target hidden states': 'llama_set_embeddings_nextn(ctx_tgt, true' in texts['spec'],
        'llama-server builds draft-specific model params': 'common_params params_dft = common_base_params_to_speculative(params_base)' in texts['server'],
        'llama-server initializes independent MTP context': 'common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt)' in texts['server'],
    }
    return rules


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('llama_cpp', type=Path)
    args = p.parse_args()
    try:
        rules = audit(args.llama_cpp)
    except (OSError, UnicodeError) as e:
        sys.exit(f'ERROR: source files unavailable: {e}')
    print(json.dumps({'source': str(args.llama_cpp.resolve()), 'checks': rules, 'pass': all(rules.values())}, indent=2))
    if not all(rules.values()):
        sys.exit('FAIL: upstream has changed; audit code rather than assuming device separation works')

if __name__ == '__main__':
    main()
