// SPDX-License-Identifier: Apache-2.0
// Exercise the real BF16 hidden-state graph path used by DSpine-over-RPC.
// The previous F32-only bridge test could not catch CUDA's F32-only RMSNorm.
#include "ggml.h"
#include "ggml-backend.h"
#include "dspine-bridge.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void verify(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
static ggml_tensor * in1(ggml_context *ctx, ggml_type typ, int64_t n) {
    auto *t=ggml_new_tensor_1d(ctx,typ,n);
    ggml_set_input(t);
    return t;
}
static ggml_tensor * in2(ggml_context *ctx, ggml_type typ, int64_t n0, int64_t n1) {
    auto *t=ggml_new_tensor_2d(ctx,typ,n0,n1);
    ggml_set_input(t);
    return t;
}
static void put_bf16(ggml_tensor *t, float value) {
    verify(t->type == GGML_TYPE_BF16 && t->buffer, "BF16 input buffer missing");
    std::vector<ggml_bf16_t> values((size_t)ggml_nelements(t),ggml_fp32_to_bf16(value));
    ggml_backend_tensor_set(t,values.data(),0,values.size()*sizeof(ggml_bf16_t));
}
static void put_f32(ggml_tensor *t, const std::vector<float> &data) {
    verify(t->type == GGML_TYPE_F32 && t->buffer && ggml_nelements(t) == (int64_t)data.size(),
           "F32 input buffer missing");
    ggml_backend_tensor_set(t,data.data(),0,data.size()*sizeof(float));
}
static void put_i32(ggml_tensor *t, const std::vector<int32_t> &data) {
    verify(t->type == GGML_TYPE_I32 && t->buffer && ggml_nelements(t) == (int64_t)data.size(),
           "I32 input buffer missing");
    ggml_backend_tensor_set(t,data.data(),0,data.size()*sizeof(int32_t));
}

int main() {
    try {
        ggml_init_params params={};
        params.mem_size=8*1024*1024;
        params.no_alloc=true;
        ggml_context *ctx=ggml_init(params);
        verify(ctx!=nullptr,"BF16 GGML context initialization failed");
        constexpr int64_t H=4,C=4,R=2,V=8,T=2;
        llama_dspine::bridge_weights w;
        w.codebook=in2(ctx,GGML_TYPE_BF16,C,V);
        w.read_w=in2(ctx,GGML_TYPE_BF16,H,C);
        w.gate_w=in2(ctx,GGML_TYPE_BF16,C,R);
        w.sender_w=in2(ctx,GGML_TYPE_BF16,C,R);
        w.value_w=in2(ctx,GGML_TYPE_BF16,C,R);
        w.gate_bias=in1(ctx,GGML_TYPE_BF16,R);
        w.write_w=in2(ctx,GGML_TYPE_BF16,R,H);
        w.next_norm_w=in1(ctx,GGML_TYPE_BF16,H);
        llama_dspine::bridge_inputs in;
        in.hidden=in2(ctx,GGML_TYPE_BF16,H,T);
        in.anchor_token_ids=in1(ctx,GGML_TYPE_I32,T);
        in.predecessor_ids=in1(ctx,GGML_TYPE_I32,T);
        in.anchor_mask=in2(ctx,GGML_TYPE_F32,1,T);
        in.edge_mask=in2(ctx,GGML_TYPE_F32,1,T);
        in.eps=in1(ctx,GGML_TYPE_F32,1);

        auto o=llama_dspine::build_bridge(ctx,in,w,(int)T,1e-6f);
        verify(o.residual->type == GGML_TYPE_BF16 && o.normalized->type == GGML_TYPE_BF16,
               "DSpine BF16 round-trip storage contract violated");
        ggml_set_output(o.residual);
        ggml_set_output(o.normalized);
        ggml_cgraph *graph=ggml_new_graph(ctx);
        ggml_build_forward_expand(graph,o.residual);
        ggml_build_forward_expand(graph,o.normalized);
        int n_rms=0;
        for(int i=0;i<ggml_graph_n_nodes(graph);i++) {
            const auto *node=ggml_graph_node(graph,i);
            if(node->op == GGML_OP_RMS_NORM) {
                ++n_rms;
                verify(node->src[0] && node->src[0]->type == GGML_TYPE_F32 &&
                       node->type == GGML_TYPE_F32,
                       "CUDA would abort: GGML RMSNorm has non-F32 input/output");
            }
        }
        verify(n_rms == 1,"DSpine BF16 bridge must contain exactly one F32 RMSNorm");

        ggml_backend_t backend=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
        verify(backend!=nullptr,"CPU backend init failed");
        auto *sched=ggml_backend_sched_new(&backend,nullptr,1,GGML_DEFAULT_GRAPH_SIZE,false,true);
        verify(sched!=nullptr,"GGML scheduler init failed");
        verify(ggml_backend_sched_alloc_graph(sched,graph),"BF16 bridge allocation failed");
        for(ggml_tensor *t : {w.codebook,w.read_w,w.gate_w,w.sender_w,
                              w.value_w,w.gate_bias,w.write_w})put_bf16(t,0.0f);
        put_bf16(w.next_norm_w,1.0f);
        put_bf16(in.hidden,1.0f);
        put_i32(in.anchor_token_ids,{0,0});
        put_i32(in.predecessor_ids,{0,0});
        put_f32(in.anchor_mask,{1.0f,0.0f});
        put_f32(in.edge_mask,{0.0f,1.0f});
        put_f32(in.eps,{1e-6f});
        verify(ggml_backend_sched_graph_compute(sched,graph)==GGML_STATUS_SUCCESS,
               "BF16 bridge compute failed");
        std::vector<ggml_bf16_t> output((size_t)(H*T));
        ggml_backend_tensor_get(o.normalized,output.data(),0,output.size()*sizeof(ggml_bf16_t));
        for(const auto &v:output) {
            verify(std::fabs(ggml_bf16_to_fp32(v)-1.0f)<0.04f,
                   "BF16 bridge identity normalization produced unexpected output");
        }
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_free(ctx);
        std::puts("DSpine BF16 bridge F32-CUDA-compatible RMSNorm and round-trip: PASS");
        return 0;
    } catch(const std::exception &e) {
        std::fprintf(stderr,"DSpine BF16 bridge FAILED: %s\n",e.what());
        return 1;
    }
}
