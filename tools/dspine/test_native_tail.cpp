// SPDX-License-Identifier: Apache-2.0
// Standalone CPU contract for the entire DSpine greedy proposal *tail*;
// not a complete 5-layer transformer or a speculative-verification engine.
#include "ggml.h"
#include "ggml-backend.h"
#include "dspine-tail.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

static void require(bool v, const char * why) { if (!v) throw std::runtime_error(why); }
static ggml_tensor * tensor(ggml_context * ctx, ggml_type typ, int64_t x, int64_t y=1) {
    auto * t = ggml_new_tensor_2d(ctx, typ, x, y);
    ggml_set_input(t);
    return t;
}
template<class T> static void put(ggml_tensor * t, const std::vector<T> & v) {
    require(t->buffer && ggml_nelements(t) == (int64_t)v.size(), "bad input/leaf allocation");
    ggml_backend_tensor_set(t, v.data(), 0, v.size()*sizeof(T));
}
int main() {
    try {
        constexpr int64_t H=4, C=4, R=2, V=8, T=3, K=2;
        ggml_init_params p = {};
        p.mem_size = 4 * 1024 * 1024;
        p.no_alloc = true;
        auto * ctx = ggml_init(p);
        require(ctx!=nullptr, "GGML metadata context failed");

        llama_dspine::bridge_weights bridge;
        bridge.codebook = tensor(ctx,GGML_TYPE_F32,C,V);
        bridge.read_w   = tensor(ctx,GGML_TYPE_F32,H,C);
        bridge.gate_w   = tensor(ctx,GGML_TYPE_F32,C,R);
        bridge.sender_w = tensor(ctx,GGML_TYPE_F32,C,R);
        bridge.value_w  = tensor(ctx,GGML_TYPE_F32,C,R);
        bridge.gate_bias = tensor(ctx,GGML_TYPE_F32,R);
        bridge.write_w = tensor(ctx,GGML_TYPE_F32,R,H);
        bridge.next_norm_w = tensor(ctx,GGML_TYPE_F32,H);

        llama_dspine::bridge_inputs input;
        input.hidden = tensor(ctx, GGML_TYPE_F32, H, T);
        input.anchor_token_ids = tensor(ctx, GGML_TYPE_I32, T);
        input.predecessor_ids = tensor(ctx, GGML_TYPE_I32, T);
        input.anchor_mask = tensor(ctx, GGML_TYPE_F32, 1,T);
        input.edge_mask = tensor(ctx, GGML_TYPE_F32, 1,T);
        input.eps = tensor(ctx, GGML_TYPE_F32,1);

        llama_dspine::refine_weights refine;
        refine.lm_head = tensor(ctx,GGML_TYPE_F32,H,V);
        refine.norm_w  = tensor(ctx,GGML_TYPE_F32,H);
        refine.tg = tensor(ctx,GGML_TYPE_F32,R,V);
        refine.tsv = tensor(ctx,GGML_TYPE_F32,2*R,V);
        refine.gate_bias = bridge.gate_bias;

        auto tail = llama_dspine::build_greedy_tail(ctx, input, bridge, refine,T,K,1e-6f);
        ggml_set_output(tail.candidate_ids);
        ggml_set_output(tail.predecessor_ids);
        ggml_set_output(tail.transitions);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph,tail.candidate_ids);
        ggml_build_forward_expand(graph,tail.predecessor_ids);
        ggml_build_forward_expand(graph,tail.transitions);

        ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
        require(backend!=nullptr,"CPU backend init failed");
        auto * sched = ggml_backend_sched_new(&backend,nullptr,1,GGML_DEFAULT_GRAPH_SIZE,false,true);
        require(sched!=nullptr,"GGML backend scheduler init failed");
        require(ggml_backend_sched_alloc_graph(sched,graph),"tail graph allocation failed");

        auto zeros=[](ggml_tensor * t) {
            put(t,std::vector<float>((size_t)ggml_nelements(t),0.f));
        };
        zeros(bridge.codebook);
        zeros(bridge.read_w);
        zeros(bridge.gate_w);
        zeros(bridge.sender_w);
        zeros(bridge.value_w);
        zeros(bridge.gate_bias);
        zeros(bridge.write_w);
        put(bridge.next_norm_w,std::vector<float>(H,1));
        put(input.hidden,std::vector<float>{1,0,0,0, 1,0,0,0, 0,1,0,0});
        put(input.anchor_token_ids,std::vector<int32_t>{1,0,0});
        put(input.predecessor_ids,std::vector<int32_t>{0,0,1});
        put(input.anchor_mask,std::vector<float>{1,0,0});
        put(input.edge_mask,std::vector<float>{0,1,1});
        put(input.eps,std::vector<float>{1e-6f});
        std::vector<float> lm(H*V,0.f);
        lm[H*2+0]=4.f;
        lm[H*3+0]=3.f;
        lm[H*4+1]=4.f;
        lm[H*5+1]=3.f;
        put(refine.lm_head,lm);
        put(refine.norm_w,std::vector<float>(H,1));
        zeros(refine.tg);
        zeros(refine.tsv);

        require(ggml_backend_sched_graph_compute(sched,graph)==GGML_STATUS_SUCCESS,
                "native DSpine proposal tail graph compute failed");
        std::vector<int32_t> candidates(K*(T-1)), predecessors(K*(T-1));
        std::vector<int32_t> transitions(K*(T-1));
        ggml_backend_tensor_get(tail.candidate_ids,candidates.data(),0,candidates.size()*sizeof(int32_t));
        ggml_backend_tensor_get(tail.predecessor_ids,predecessors.data(),0,predecessors.size()*sizeof(int32_t));
        ggml_backend_tensor_get(tail.transitions,transitions.data(),0,transitions.size()*sizeof(int32_t));
        require(std::find(candidates.begin(),candidates.begin()+K,2)!=candidates.begin()+K &&
                std::find(candidates.begin(),candidates.begin()+K,3)!=candidates.begin()+K,
                "step 1 top-K candidates incorrect");
        require(std::find(candidates.begin()+K,candidates.end(),4)!=candidates.end() &&
                std::find(candidates.begin()+K,candidates.end(),5)!=candidates.end(),
                "step 2 top-K candidates incorrect");
        require(predecessors[0]==1 && predecessors[1]==1 &&
                predecessors[2]==candidates[0] && predecessors[3]==candidates[1],
                "DSpine predecessor candidate lattice shifted incorrectly");
        const auto generated = llama_dspine::serial_chain(transitions,candidates,T,K);
        require(generated == std::vector<int32_t>({2,4}),
                "DSpine greedy serial chain did not pick expected best targets");
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_free(ctx);
        std::puts("DSpine native GGML bridge + top-K + serial refinement tail: PASS");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr,"DSpine tail test FAILED: %s\n",e.what());
        return 1;
    }
}
