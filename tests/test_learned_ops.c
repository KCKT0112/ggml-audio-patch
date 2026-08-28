// Smoke tests for the learned-ops patch (ggml v0.19.0 base).
// Verifies, against hand-rolled references:
//   1. ggml_conv_1d_fast_1d_im2col == ggml_conv_1d        (CPU reference kernels)
//   2. ggml_conv_transpose_1d_ext  == naive reference     (output_padding, groups, padding)
//   3. ggml_scatter_elements       == naive reference     (overwrite / add, axis 0/1)
//   4. ggml_rel_pos_bias           == naive reference
//
// Usage: test_learned_ops [cpu|vk]
// (vk builds link ggml-vulkan and run every case on device 0; cases the
//  backend reports as unsupported are skipped - on Vulkan that is convT
//  with p0 != 0 / d0 != 1 and scatter-add without VK_EXT_shader_atomic_float.)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef USE_VULKAN
// provided by ggml-vulkan.dll (ggml.h GGML_API decls live in the backend lib)
ggml_backend_t             ggml_backend_vk_init(size_t dev_num);
ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
#endif

static int failures = 0;
static const char * g_backend = "cpu";

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: " __VA_ARGS__); printf("  (%s:%d)\n", __FILE__, __LINE__); \
        failures++; \
    } \
} while (0)

static bool vec_close(const float * a, const float * b, int n, float tol) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > tol) {
            printf("    mismatch at %d: %f vs %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

static bool find_op_node(const struct ggml_tensor * t, enum ggml_op op, int depth) {
    if (t == NULL || depth > 12) return false;
    if (t->op == op) return true;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (t->src[i] && find_op_node(t->src[i], op, depth+1)) return true;
    }
    return false;
}

// per-test-case execution context
struct tctx {
    struct ggml_context * ctx;
    ggml_backend_t backend;
    ggml_backend_buffer_type_t buft;
    ggml_gallocr_t galloc;
    struct ggml_cgraph * gf;
};

static void tctx_begin(struct tctx * t) {
    size_t buf_size = 32u*1024u*1024u;
    struct ggml_init_params ip = { buf_size, NULL, /* no_alloc */ true };
    t->ctx = ggml_init(ip);
    t->backend = NULL;
    t->buft = NULL;
    t->galloc = NULL;
    t->gf = NULL;
#ifdef USE_VULKAN
    if (strcmp(g_backend, "vk") == 0) {
        t->backend = ggml_backend_vk_init(0);
        t->buft = ggml_backend_vk_buffer_type(0);
        return;
    }
#endif
    t->backend = ggml_backend_cpu_init();
    t->buft = ggml_backend_cpu_buffer_type();
}

// build the forward graph for `result` (plus optional second root `result2`)
// allocate intermediates on the test device and prepare for compute
static bool tctx_alloc_graph(struct tctx * t, struct ggml_tensor * result, struct ggml_tensor * result2) {
    if (!ggml_backend_supports_op(t->backend, result)) {
        printf("  SKIP: %s does not support this op shape\n", g_backend);
        return false;
    }
    t->gf = ggml_new_graph(t->ctx);
    ggml_build_forward_expand(t->gf, result);
    if (result2) {
        ggml_build_forward_expand(t->gf, result2);
    }
    t->galloc = ggml_gallocr_new(t->buft);
    if (!ggml_gallocr_alloc_graph(t->galloc, t->gf)) {
        printf("  SKIP: galloc failed\n");
        return false;
    }
    return true;
}

static void tctx_upload(struct tctx * t, struct ggml_tensor * tensor, const void * host) {
    ggml_backend_tensor_set(tensor, host, 0, ggml_nbytes(tensor));
}

static void tctx_download(struct tctx * t, const struct ggml_tensor * tensor, void * host) {
    ggml_backend_tensor_get(tensor, host, 0, ggml_nbytes(tensor));
}

static void tctx_compute(struct tctx * t) {
    ggml_backend_graph_compute(t->backend, t->gf);
}

static void tctx_end(struct tctx * t) {
    if (t->galloc) ggml_gallocr_free(t->galloc);
    if (t->backend) ggml_backend_free(t->backend);
    ggml_free(t->ctx);
}

// ---------- 1. conv_1d vs conv_1d_fast_1d_im2col ----------

static void test_conv_1d_parity(void) {
    printf("[test] conv_1d vs conv_1d_fast_1d_im2col parity\n");

    struct {
        int N, OC, IC, K, s0, p0, d0;
    } cases[] = {
        {1, 3, 2, 3, 1, 1, 1},
        {2, 4, 3, 5, 2, 2, 1},
        {1, 2, 1, 7, 3, 1, 1},
        {1, 4, 2, 3, 2, 0, 2},   // dilated
        {3, 8, 4, 4, 2, 1, 1},
        {1, 1, 1, 1, 1, 0, 1},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int N = cases[c].N, OC = cases[c].OC, IC = cases[c].IC;
        const int K = cases[c].K, s0 = cases[c].s0, p0 = cases[c].p0, d0 = cases[c].d0;
        const int L = 23 + (int)c;
        const int OL = (L + 2*p0 - d0*(K-1) - 1)/s0 + 1;

        struct tctx t;
        tctx_begin(&t);

        // ggml conv_1d kernel layout: [K, IC, OC] (NOT PyTorch [OC, IC, K])
        struct ggml_tensor * a = ggml_new_tensor_3d(t.ctx, GGML_TYPE_F32, K, IC, OC);
        struct ggml_tensor * b = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, L, IC);
        ggml_set_name(a, "a");
        ggml_set_name(b, "b");

        float * pa = (float *)malloc(ggml_nbytes(a));
        float * pb = (float *)malloc(ggml_nbytes(b));
        for (int i = 0; i < (int)ggml_nelements(a); i++) pa[i] = (float)((i*7 + c) % 13) * 0.25f - 1.0f;
        for (int i = 0; i < (int)ggml_nelements(b); i++) pb[i] = (float)((i*11 + (int)c*3) % 17) * 0.2f - 1.5f;

        struct ggml_tensor * r1 = ggml_conv_1d(t.ctx, a, b, s0, p0, d0);
        struct ggml_tensor * r2 = ggml_conv_1d_fast_1d_im2col(t.ctx, a, b, s0, p0, d0);
        CHECK(r1->ne[0] == r2->ne[0] && r1->ne[1] == r2->ne[1] && r1->ne[2] == r2->ne[2],
              "shape mismatch (case %zu)", c);
        CHECK(OL == (int)r1->ne[0], "unexpected output length (case %zu)", c);
        CHECK(find_op_node(r2, GGML_OP_IM2COL_FAST_1D, 0), "im2col not tagged FAST_1D (case %zu)", c);

        if (!tctx_alloc_graph(&t, r1, r2)) { free(pa); free(pb); tctx_end(&t); continue; }
        tctx_upload(&t, a, pa);
        tctx_upload(&t, b, pb);
        tctx_compute(&t);

        float * o1 = (float *)malloc(ggml_nbytes(r1));
        float * o2 = (float *)malloc(ggml_nbytes(r2));
        tctx_download(&t, r1, o1);
        tctx_download(&t, r2, o2);
        CHECK(vec_close(o1, o2, (int)ggml_nelements(r1), 1e-3f), "outputs differ (case %zu: N%d OC%d IC%d K%d s%d p%d d%d)",
              c, N, OC, IC, K, s0, p0, d0);

        free(pa); free(pb); free(o1); free(o2);
        tctx_end(&t);
    }
    printf("  done (%d failures so far)\n", failures);
}

// ---------- 2. conv_transpose_1d_ext ----------

static void test_conv_transpose_1d_ext(void) {
    printf("[test] conv_transpose_1d_ext (output_padding / groups / padding)\n");

    struct {
        int L, Cin, Cout, K, s0, p0, op0, g0;
    } cases[] = {
        {5, 2, 4, 3, 2, 0, 0, 1},   // plain
        {5, 2, 4, 3, 2, 0, 2, 1},   // output_padding
        {6, 4, 4, 3, 2, 1, 1, 1},   // padding
        {6, 4, 4, 3, 2, 1, 0, 2},   // groups == 2
        {7, 6, 3, 4, 2, 1, 1, 3},   // groups == 3
        {4, 3, 3, 3, 1, 0, 1, 3},   // groups == Cin (depthwise-ish)
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int L = cases[c].L, Cin = cases[c].Cin, Cout = cases[c].Cout;
        const int K = cases[c].K, s0 = cases[c].s0, p0 = cases[c].p0;
        const int op0 = cases[c].op0, g0 = cases[c].g0;
        const int Cin_g = Cin / g0, Cout_pg = Cout / g0;
        const int OL = (L - 1)*s0 - 2*p0 + (K - 1) + op0 + 1;

        struct tctx t;
        tctx_begin(&t);

        // a: [K, Cout_pg, Cin_g], b: [L, Cin]
        struct ggml_tensor * a = ggml_new_tensor_3d(t.ctx, GGML_TYPE_F32, K, Cout_pg, Cin_g);
        struct ggml_tensor * b = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, L, Cin);

        float * pa = (float *)malloc(ggml_nbytes(a));
        float * pb = (float *)malloc(ggml_nbytes(b));
        for (int i = 0; i < (int)ggml_nelements(a); i++) pa[i] = (float)((i*5 + (int)c) % 9) * 0.3f - 1.0f;
        for (int i = 0; i < (int)ggml_nelements(b); i++) pb[i] = (float)((i*3 + 1) % 11) * 0.25f - 1.0f;

        struct ggml_tensor * r = ggml_conv_transpose_1d_ext(t.ctx, a, b, s0, p0, 1, op0, g0);
        CHECK((int)r->ne[0] == OL && (int)r->ne[1] == Cout, "shape (case %zu): got [%d,%d] want [%d,%d]",
              c, (int)r->ne[0], (int)r->ne[1], OL, Cout);

        if (!tctx_alloc_graph(&t, r, NULL)) { free(pa); free(pb); tctx_end(&t); continue; }
        tctx_upload(&t, a, pa);
        tctx_upload(&t, b, pb);
        tctx_compute(&t);

        float * out = (float *)malloc(ggml_nbytes(r));
        tctx_download(&t, r, out);

        // naive reference
        float * ref = (float *)calloc((size_t)OL * Cout, sizeof(float));
        for (int oc = 0; oc < Cout; oc++) {
            const int g = oc / Cout_pg;
            const int oc_pg = oc % Cout_pg;
            for (int i10 = 0; i10 < L; i10++) {
                for (int k = 0; k < K; k++) {
                    const int64_t o = (int64_t)i10*s0 + k - p0;
                    if (o < 0 || o >= OL) continue;
                    float acc = 0.0f;
                    for (int cg = 0; cg < Cin_g; cg++) {
                        const float w = pa[k + oc_pg*K + cg*K*Cout_pg];
                        const float x = pb[i10 + (g*Cin_g + cg)*L];
                        acc += w * x;
                    }
                    ref[oc*OL + o] += acc;
                }
            }
        }
        CHECK(vec_close(out, ref, OL*Cout, 1e-3f), "convT output differs (case %zu: L%d Cin%d Cout%d K%d s%d p%d op%d g%d)",
              c, L, Cin, Cout, K, s0, p0, op0, g0);
        free(ref);

        free(pa); free(pb); free(out);
        tctx_end(&t);
    }
    printf("  done (%d failures so far)\n", failures);
}

// ---------- 3. scatter_elements ----------

static void test_scatter_elements(void) {
    printf("[test] scatter_elements (overwrite / add, axis 0/1)\n");

    // 2D case, axis 1
    {
        struct tctx t;
        tctx_begin(&t);

        const int d0 = 4, d1 = 6, u1 = 3;
        struct ggml_tensor * data = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, d0, d1);
        struct ggml_tensor * upd  = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, d0, u1);
        struct ggml_tensor * idx  = ggml_new_tensor_2d(t.ctx, GGML_TYPE_I32, d0, u1);

        float * pd = (float *)malloc(ggml_nbytes(data));
        float * pu = (float *)malloc(ggml_nbytes(upd));
        int32_t * pi = (int32_t *)malloc(ggml_nbytes(idx));
        for (int i = 0; i < d0*d1; i++) pd[i] = (float)i * 0.5f;
        for (int i = 0; i < d0*u1; i++) pu[i] = (float)(100 + i);
        for (int i = 0; i < d0*u1; i++) pi[i] = (i * 5 + 2) % d1;

        struct ggml_tensor * r0 = ggml_scatter_elements(t.ctx, data, upd, idx, 0, 1);
        struct ggml_tensor * r1 = ggml_scatter_elements(t.ctx, data, upd, idx, 1, 1);

        if (!tctx_alloc_graph(&t, r0, r1)) { free(pd); free(pu); free(pi); tctx_end(&t); return; }
        tctx_upload(&t, data, pd);
        tctx_upload(&t, upd, pu);
        tctx_upload(&t, idx, pi);
        tctx_compute(&t);

        float * o0 = (float *)malloc(ggml_nbytes(r0));
        float * o1 = (float *)malloc(ggml_nbytes(r1));
        tctx_download(&t, r0, o0);
        tctx_download(&t, r1, o1);

        float * ref0 = (float *)malloc(sizeof(float)*d0*d1);
        float * ref1 = (float *)malloc(sizeof(float)*d0*d1);
        memcpy(ref0, pd, sizeof(float)*d0*d1);
        memcpy(ref1, pd, sizeof(float)*d0*d1);
        for (int j = 0; j < u1; j++) {
            for (int i = 0; i < d0; i++) {
                const int flat = j*d0 + i;
                const int tt = pi[flat];
                if (tt < 0) continue;
                ref0[i + tt*d0] = pu[flat];
                ref1[i + tt*d0] += pu[flat];
            }
        }

        CHECK(vec_close(o0, ref0, d0*d1, 0.0f), "scatter overwrite (axis 1) differs");
        CHECK(vec_close(o1, ref1, d0*d1, 0.0f), "scatter add (axis 1) differs");
        free(pd); free(pu); free(pi); free(o0); free(o1); free(ref0); free(ref1);
        tctx_end(&t);
    }

    // 1D case, axis 0, duplicate indices
    {
        struct tctx t;
        tctx_begin(&t);

        const int n = 10, m = 4;
        struct ggml_tensor * data = ggml_new_tensor_1d(t.ctx, GGML_TYPE_F32, n);
        struct ggml_tensor * upd  = ggml_new_tensor_1d(t.ctx, GGML_TYPE_F32, m);
        struct ggml_tensor * idx  = ggml_new_tensor_1d(t.ctx, GGML_TYPE_I32, m);

        float * pd = (float *)malloc(ggml_nbytes(data));
        float * pu = (float *)malloc(ggml_nbytes(upd));
        int32_t * pi = (int32_t *)malloc(ggml_nbytes(idx));
        for (int i = 0; i < n; i++) pd[i] = (float)i;
        for (int i = 0; i < m; i++) pu[i] = (float)(-i);
        pi[0] = 3; pi[1] = 0; pi[2] = 9; pi[3] = 3;   // duplicate on purpose

        struct ggml_tensor * r1 = ggml_scatter_elements(t.ctx, data, upd, idx, 1, 0);

        if (!tctx_alloc_graph(&t, r1, NULL)) { free(pd); free(pu); free(pi); tctx_end(&t); return; }
        tctx_upload(&t, data, pd);
        tctx_upload(&t, upd, pu);
        tctx_upload(&t, idx, pi);
        tctx_compute(&t);

        float * o1 = (float *)malloc(ggml_nbytes(r1));
        tctx_download(&t, r1, o1);

        float ref[10];
        for (int i = 0; i < n; i++) ref[i] = (float)i;
        for (int i = 0; i < m; i++) ref[pi[i]] += pu[i];

        CHECK(vec_close(o1, ref, n, 0.0f), "scatter add (axis 0, duplicates) differs");

        free(pd); free(pu); free(pi); free(o1);
        tctx_end(&t);
    }
    printf("  done (%d failures so far)\n", failures);
}

// ---------- 4. rel_pos_bias ----------

static void test_rel_pos_bias(void) {
    printf("[test] rel_pos_bias vs naive reference\n");

    struct { int C, H, W, B; } cases[] = {
        {2, 3, 4, 1},
        {3, 2, 2, 2},
        {1, 5, 3, 1},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int C = cases[c].C, H = cases[c].H, W = cases[c].W, B = cases[c].B;
        const int HW = H*W, rel_h = 2*H - 1, rel_w = 2*W - 1, Ws = rel_h + rel_w;

        struct tctx t;
        tctx_begin(&t);

        struct ggml_tensor * x    = ggml_new_tensor_3d(t.ctx, GGML_TYPE_F32, C, HW, B);
        struct ggml_tensor * wcat = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, Ws, C);

        float * px = (float *)malloc(ggml_nbytes(x));
        float * pw = (float *)malloc(ggml_nbytes(wcat));
        for (int i = 0; i < C*HW*B; i++) px[i] = (float)((i*3 + (int)c) % 7) * 0.4f - 1.0f;
        for (int i = 0; i < C*Ws; i++)  pw[i] = (float)((i*5 + 1) % 11) * 0.3f - 1.2f;

        struct ggml_tensor * r = ggml_rel_pos_bias(t.ctx, x, wcat, H, W);
        CHECK((int)r->ne[0] == HW && (int)r->ne[1] == HW && (int)r->ne[2] == B, "shape (case %zu)", c);

        if (!tctx_alloc_graph(&t, r, NULL)) { free(px); free(pw); tctx_end(&t); continue; }
        tctx_upload(&t, x, px);
        tctx_upload(&t, wcat, pw);
        tctx_compute(&t);

        float * out = (float *)malloc(ggml_nbytes(r));
        tctx_download(&t, r, out);

        // reference (ggml col-major): out[k + q*HW + b*HW*HW] =
        //   sum_c x[c + q*C + b*C*HW] * w[c*Ws + r_h]  +
        //   sum_c x[c + (wq*H+hq)*C + b*C*HW] * w[c*Ws + rel_h + r_w]
        float * ref = (float *)calloc((size_t)HW*HW*B, sizeof(float));
        for (int b = 0; b < B; b++) {
            for (int q = 0; q < HW; q++) {
                const int hq = q / W, wq = q % W;
                for (int hk = 0; hk < H; hk++) {
                    for (int wk = 0; wk < W; wk++) {
                        const int k = hk*W + wk;
                        const int r_h = hq - hk + H - 1;
                        const int r_w = wq - wk + W - 1;
                        float s = 0.0f;
                        for (int ci = 0; ci < C; ci++) {
                            s += px[ci + q*C + b*C*HW] * pw[ci*Ws + r_h];
                            s += px[ci + (wq*H + hq)*C + b*C*HW] * pw[ci*Ws + rel_h + r_w];
                        }
                        ref[k + (size_t)q*HW + (size_t)b*HW*HW] = s;
                    }
                }
            }
        }
        CHECK(vec_close(out, ref, HW*HW*B, 1e-3f), "rel_pos_bias differs (case %zu: C%d H%d W%d B%d)", c, C, H, W, B);
        free(ref);

        free(px); free(pw); free(out);
        tctx_end(&t);
    }
    printf("  done (%d failures so far)\n", failures);
}

int main(int argc, char ** argv) {
    if (argc > 1) {
        g_backend = argv[1];
    }
    printf("== learned-ops smoke tests (backend: %s) ==\n", g_backend);

    test_conv_1d_parity();
    test_conv_transpose_1d_ext();
    test_scatter_elements();
    test_rel_pos_bias();

    if (failures == 0) {
        printf("\nALL PASSED\n");
        return 0;
    }
    printf("\n%d FAILURES\n", failures);
    return 1;
}
