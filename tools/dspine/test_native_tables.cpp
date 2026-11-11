// SPDX-License-Identifier: Apache-2.0
// Native BF16 table precompute contract with exact-representable toy weights.
#include "ggml.h"
#include "ggml-backend.h"
#include "dspine-tables.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

static void expect(bool v,const char * why){if(!v)throw std::runtime_error(why);}
static ggml_tensor * input(ggml_context * ctx,int64_t x,int64_t y=1){
    auto * t=ggml_new_tensor_2d(ctx,GGML_TYPE_BF16,x,y);
    ggml_set_input(t);
    return t;
}
static void assign(ggml_tensor * t,const std::vector<float> & data){
    expect(t->buffer && ggml_nelements(t)==(int64_t)data.size(),"unallocated input/wrong shape");
    std::vector<ggml_bf16_t> bf(data.size());
    for(size_t i=0;i<data.size();i++)bf[i]=ggml_fp32_to_bf16(data[i]);
    ggml_backend_tensor_set(t,bf.data(),0,bf.size()*sizeof(ggml_bf16_t));
}
int main(){
    try{
        constexpr int64_t H=4,V=8193,R=2,C=4; // cross the 8192-row GPU tile boundary
        ggml_init_params params={};
        params.mem_size=4*1024*1024;params.no_alloc=true;
        auto * ctx=ggml_init(params);
        expect(ctx!=nullptr,"GGML init failed");
        auto * head=input(ctx,H,V);
        auto * norm=input(ctx,H);
        auto * write=input(ctx,R,H);
        auto * codebook=input(ctx,C,V);
        auto * gate=input(ctx,C,R);
        auto * value=input(ctx,C,R);
        auto tables=llama_dspine::build_serial_tables(ctx,head,norm,write,codebook,gate,value);
        ggml_set_output(tables.tg);
        ggml_set_output(tables.tsv);
        ggml_cgraph * graph=ggml_new_graph(ctx);
        ggml_build_forward_expand(graph,tables.tg);
        ggml_build_forward_expand(graph,tables.tsv);
        ggml_backend_t backend=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
        expect(backend!=nullptr,"CPU backend init failed");
        auto * sched=ggml_backend_sched_new(&backend,nullptr,1,GGML_DEFAULT_GRAPH_SIZE,false,true);
        expect(sched!=nullptr,"scheduler init failed");
        expect(ggml_backend_sched_alloc_graph(sched,graph),"table allocation failed");
        std::vector<float> h(H*V,0), w(R*H,0), cb(C*V,0), g(C*R,0), v(C*R,0);
        h[H*1+0]=2;h[H*1+1]=3; // target token1
        h[H*(V-1)+0]=2;h[H*(V-1)+1]=3; // last token lives in tile 2
        w[R*0+0]=1;w[R*1+1]=1;
        cb[C*1+0]=1;cb[C*1+1]=2;
        cb[C*(V-1)+0]=1;cb[C*(V-1)+1]=2;
        g[C*0+0]=1;g[C*1+1]=1;
        v[C*1+1]=1;
        assign(head,h);assign(norm,{1,1,1,1});assign(write,w);
        assign(codebook,cb);assign(gate,g);assign(value,v);
        expect(ggml_backend_sched_graph_compute(sched,graph)==GGML_STATUS_SUCCESS,
               "table graph execution failed");
        std::vector<ggml_bf16_t> tg(R*V),tsv(2*R*V);
        ggml_backend_tensor_get(tables.tg,tg.data(),0,tg.size()*sizeof(ggml_bf16_t));
        ggml_backend_tensor_get(tables.tsv,tsv.data(),0,tsv.size()*sizeof(ggml_bf16_t));
        const float expected_tg[]={2.f,3.f};
        const float expected_tsv[]={2.f,4.f,0.f,4.f};
        for(int i=0;i<2;i++)
            expect(std::fabs(ggml_bf16_to_fp32(tg[(size_t)(R+i)])-expected_tg[i])<0.01f,
                   "TG table numerical mismatch");
        for(int i=0;i<4;i++)
            expect(std::fabs(ggml_bf16_to_fp32(tsv[(size_t)(2*R+i)])-expected_tsv[i])<0.01f,
                   "TSV table numerical mismatch");
        // Both TG and TSV must preserve the same token order when the
        // vocabulary is split into a full 8192-row tile and a partial tile.
        for(int i=0;i<2;i++)
            expect(std::fabs(ggml_bf16_to_fp32(tg[(size_t)(R*(V-1)+i)])-expected_tg[i])<0.01f,
                   "TG table chunk boundary corrupted token order");
        for(int i=0;i<4;i++)
            expect(std::fabs(ggml_bf16_to_fp32(tsv[(size_t)(2*R*(V-1)+i)])-expected_tsv[i])<0.01f,
                   "TSV table chunk boundary corrupted token order");
        ggml_backend_sched_free(sched);
        ggml_backend_free(backend);
        ggml_free(ctx);
        std::puts("DSpine BF16 sender TG/TSV precomputation: PASS");
        return 0;
    }catch(const std::exception & e){
        std::fprintf(stderr,"DSpine BF16 table test FAILED: %s\n",e.what());
        return 1;
    }
}
