// SPDX-License-Identifier: MIT
// Regression cases shared with the ggml-audio-patch collection.
#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static int passed = 0, skipped = 0, scatter_checks = 0;
static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

struct Graph {
    ggml_backend_t backend;
    ggml_context * ctx;
    ggml_gallocr_t alloc;
    ggml_cgraph * graph;
    explicit Graph(ggml_backend_t b) : backend(b) {
        ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
        require(ctx != nullptr, "context allocation failed");
        alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(b));
        graph = ggml_new_graph(ctx);
    }
    ~Graph() { ggml_gallocr_free(alloc); ggml_free(ctx); }
    bool prepare(ggml_tensor * y) {
        ggml_set_output(y);
        ggml_build_forward_expand(graph, y);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (!ggml_backend_supports_op(backend, ggml_graph_node(graph, i))) {
                ++skipped;
                return false;
            }
        }
        require(ggml_gallocr_alloc_graph(alloc, graph), "graph allocation failed");
        return true;
    }
    void compare(ggml_tensor * y, const std::vector<float> & expected, bool unaligned_scratch = false) {
        if (unaligned_scratch) {
            ggml_cplan plan = ggml_graph_plan(graph, 1, nullptr);
            std::vector<uint8_t> scratch(plan.work_size + 64);
            uintptr_t aligned = (reinterpret_cast<uintptr_t>(scratch.data()) + 31) & ~uintptr_t(31);
            plan.work_data = reinterpret_cast<uint8_t *>(aligned + 16);
            require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "unaligned compute failed");
        } else {
            require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute failed");
        }
        std::vector<float> actual(expected.size());
        ggml_backend_tensor_get(y, actual.data(), 0, actual.size()*sizeof(float));
        for (size_t i = 0; i < actual.size(); ++i) {
            require(std::isfinite(actual[i]) && std::fabs(actual[i]-expected[i]) < 1e-4f, "output mismatch");
        }
        ++passed;
    }
};

static void scatter(ggml_backend_t backend, int axis, int reduction) {
    Graph g(backend);
    int64_t dims[4] = {4, 5, 4, 4}, updates_dims[4] = {4, 5, 4, 4};
    updates_dims[axis] = 7;
    auto data = ggml_new_tensor(g.ctx, GGML_TYPE_F32, 4, dims);
    auto updates = ggml_new_tensor(g.ctx, GGML_TYPE_F32, 4, updates_dims);
    auto indices = ggml_new_tensor(g.ctx, GGML_TYPE_I32, 4, updates_dims);
    auto y = ggml_scatter_elements(g.ctx, data, updates, indices, reduction, axis);
    if (!g.prepare(y)) return;
    std::vector<float> base(ggml_nelements(data)), values(ggml_nelements(updates));
    std::vector<int32_t> index(values.size());
    for (size_t i=0; i<base.size(); ++i) base[i] = (i%11)*0.125f;
    auto expected = base;
    const int32_t n = (int32_t)dims[axis];
    const int32_t cases[] = {-n, -1, 1, n, -n-1, INT32_MIN, INT32_MAX};
    for (size_t i=0; i<values.size(); ++i) {
        int64_t coord[4], rem = (int64_t)i;
        for (int d=0; d<4; ++d) { coord[d]=rem%updates_dims[d]; rem/=updates_dims[d]; }
        const int c = (int)coord[axis];
        index[i] = cases[c];
        values[i] = ((int)(i%9)-4)*0.25f;
        if (c >= 3) continue; // invalid updates must leave the base unchanged
        coord[axis] = c == 0 ? 0 : c == 1 ? n-1 : 1;
        const size_t dest = coord[0]+dims[0]*(coord[1]+dims[1]*(coord[2]+dims[2]*coord[3]));
        if (reduction) expected[dest] += values[i]; else expected[dest] = values[i];
    }
    ggml_backend_tensor_set(data, base.data(), 0, base.size()*4);
    ggml_backend_tensor_set(updates, values.data(), 0, values.size()*4);
    ggml_backend_tensor_set(indices, index.data(), 0, index.size()*4);
    g.compare(y, expected);
    ++scatter_checks;
}

static void transpose_padding(ggml_backend_t backend, int groups) {
    Graph g(backend);
    const int T=4, K=5, IC=2, OC=2, stride=3, pad=2, extra=1;
    const int OL=(T-1)*stride-2*pad+K+extra;
    auto w=ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, K, OC, IC);
    auto x=ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, T, IC*groups);
    auto y=ggml_conv_transpose_1d_ext(g.ctx, w, x, stride, pad, 1, extra, groups);
    require(y->ne[0] == OL, "transpose output length mismatch");
    if (!g.prepare(y)) return;
    std::vector<float> weights(K*OC*IC), input(T*IC*groups), expected(OL*OC*groups,0);
    for (size_t i=0;i<weights.size();++i) weights[i]=((int)(i%7)-3)*0.125f;
    for (size_t i=0;i<input.size();++i) input[i]=((int)(i%5)-2)*0.25f;
    for (int group=0;group<groups;++group)
        for (int ic=0;ic<IC;++ic) for (int t=0;t<T;++t)
            for (int oc=0;oc<OC;++oc) for (int k=0;k<K;++k) {
                int dest=t*stride-pad+k;
                if (dest>=0 && dest<OL)
                    expected[(group*OC+oc)*OL+dest] += input[(group*IC+ic)*T+t]*weights[(ic*OC+oc)*K+k];
            }
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size()*4);
    ggml_backend_tensor_set(x, input.data(), 0, input.size()*4);
    g.compare(y, expected);
}

static void direct_strides(ggml_backend_t backend, bool strided) {
    Graph g(backend);
    const int K=3, IC=2, OC=17, T=7;
    auto storage=ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, strided ? OC : K, strided ? K : IC, strided ? IC : OC);
    auto w=strided ? ggml_permute(g.ctx, storage,2,0,1,3) : storage;
    // ggml assigns result->ne[axis_i] = input->ne[i]. For [OC,K,IC],
    // (2,0,1,3) therefore produces [K,IC,OC], not (1,2,0,3).
    require(w->ne[0] == K && w->ne[1] == IC && w->ne[2] == OC, "permuted weight shape mismatch");
    auto bias_storage=ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 2*OC);
    auto bias=strided ? ggml_transpose(g.ctx,ggml_view_2d(g.ctx,bias_storage,1,OC,8,0))
                      : ggml_view_1d(g.ctx,bias_storage,OC,0);
    auto x=ggml_new_tensor_2d(g.ctx,GGML_TYPE_F32,T,IC);
    auto y=ggml_conv_direct_1d(g.ctx,w,x,bias,1,1,0.1f);
    if (!g.prepare(y)) return;
    std::vector<float> weights(K*IC*OC), input(T*IC), biases(2*OC,99), expected(T*OC,0);
    for (size_t i=0;i<input.size();++i) input[i]=((int)(i%5)-2)*0.25f;
    for (int oc=0;oc<OC;++oc) {
        biases[oc*(strided ? 2 : 1)]=(oc%3-1)*0.25f;
        for (int ic=0;ic<IC;++ic) for(int k=0;k<K;++k)
            weights[strided ? (ic*K+k)*OC+oc : (oc*IC+ic)*K+k] = (oc%5+ic-k)*0.125f;
        for (int t=0;t<T;++t) {
            float acc=(oc%3-1)*0.25f;
            for (int ic=0;ic<IC;++ic) for(int k=0;k<K;++k) {
                int it=t+k-1;
                if(it>=0 && it<T) acc += (oc%5+ic-k)*0.125f*input[ic*T+it];
            }
            expected[oc*T+t]=acc<0 ? 0.1f*acc : acc;
        }
    }
    ggml_backend_tensor_set(storage,weights.data(),0,weights.size()*4);
    ggml_backend_tensor_set(bias_storage,biases.data(),0,biases.size()*4);
    ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
    g.compare(y,expected);
    if (ggml_backend_is_cpu(backend)) g.compare(y,expected,true);
}

int main(int argc, char ** argv) {
    const char * name = argc>1 ? argv[1] : "CPU";
    ggml_backend_t backend=ggml_backend_init_by_name(name,nullptr);
    if (!backend) { std::fprintf(stderr,"Backend unavailable: %s\n",name); return 1; }
    int result=0;
    try {
        for(int axis=0;axis<4;++axis) for(int reduction=0;reduction<2;++reduction) scatter(backend,axis,reduction);
        transpose_padding(backend,1);
        transpose_padding(backend,2);
        direct_strides(backend,false);
        direct_strides(backend,true);
        require(passed>0,"no regression case executed");
        if (ggml_backend_is_cpu(backend)) require(passed == 14 && skipped == 0, "CPU must execute all regression cases");
        if (std::strncmp(name, "Vulkan", 6) == 0) require(scatter_checks >= 4, "Vulkan must execute scatter on all four axes");
        std::printf("%s: %d passed, %d unsupported\n",name,passed,skipped);
    } catch(const std::exception & e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); result=1; }
    ggml_backend_free(backend);
    return result;
}
