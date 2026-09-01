// SPDX-License-Identifier: MPL-2.0
// Smoke tests for the qvac-ops patch (applies on top of the learned-ops patch,
// ggml v0.19.0 base). Ported from tetherto/qvac-ext-ggml (speech branch, MIT).
// Verifies, against hand-rolled references:
//   1. ggml_supertonic_depthwise_1d (+_ct, +_causal_ct) == naive reference
//   2. ggml_supertonic_layer_norm_channel (+_ct)        == naive reference
//   3. ggml_supertonic_pw2_residual (+_ct)              == naive reference
//   4. ggml_supertonic_bias_gelu (+_ct)                 == add + gelu_erf chain
//   5. ggml_supertonic_edge_pad_1d (+_ct)               == naive reference
//   6. ggml_gru                                         == naive reference
//   7. ggml_zero_upsample                               == naive reference
//   8. ggml_channel_shuffle                             == naive reference
//   9. ggml_affine_prelu                                == naive reference
//  10. ggml_snake                                       == naive reference
//
// ggml tensors are column-major: element (i0, i1, ...) at i0 + i1*ne0 + ...
// Usage: test_qvac_ops [cpu|vk|metal]
//   metal requires -DUSE_METAL and a macOS build linking ggml-metal; it is
//   the harness for the optional Metal patch (docs/metal-porting.md in the
//   ggml-audio-patch repo).  With patch 3, the five Supertonic ops and Snake
//   execute on Metal; the four ops without donor kernels cleanly SKIP.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef USE_VULKAN
ggml_backend_t             ggml_backend_vk_init(size_t dev_num);
ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
#endif
#ifdef USE_METAL
ggml_backend_t ggml_backend_metal_init(void);
#endif

static int failures = 0;
static const char * g_backend = "cpu";

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: " __VA_ARGS__); printf("  (%s:%d)\n", __FILE__, __LINE__); \
        failures++; \
    } \
} while (0)

static bool vec_close_idx(const float * a, const float * b, int n, float tol) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > tol) {
            printf("    mismatch at %d: %f vs %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
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
#ifdef USE_METAL
    if (strcmp(g_backend, "metal") == 0) {
        t->backend = ggml_backend_metal_init();
        GGML_ASSERT(t->backend && "USE_METAL build but Metal initialization failed");
        t->buft = ggml_backend_get_default_buffer_type(t->backend);
        return;
    }
#endif
    t->backend = ggml_backend_cpu_init();
    t->buft = ggml_backend_cpu_buffer_type();
}

static bool tctx_alloc_graph(struct tctx * t, struct ggml_tensor * r0,
                             struct ggml_tensor * r1, struct ggml_tensor * r2) {
    if (!ggml_backend_supports_op(t->backend, r0)) {
        printf("  SKIP: %s does not support this op shape\n", g_backend);
        return false;
    }
    t->gf = ggml_new_graph(t->ctx);
    ggml_build_forward_expand(t->gf, r0);
    if (r1) ggml_build_forward_expand(t->gf, r1);
    if (r2) ggml_build_forward_expand(t->gf, r2);
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

static float frand(unsigned * seed) {
    *seed = *seed * 1103515245u + 12345u;
    return (float)((*seed >> 16) & 0x7fff) / 16384.0f - 1.0f;
}

static float gelu_erf_f(float v) {
    return 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
}

// ---------- 1. supertonic_depthwise_1d ----------
// x: [L, C] col-major (element (t,c) at t + c*L).  ct variant: [C, L].

static void test_supertonic_depthwise_1d(void) {
    printf("[test] supertonic_depthwise_1d / _ct / _causal_ct\n");

    struct { int L, C, K, dil, causal; } cases[] = {
        {11, 3, 3, 1, 0},
        {11, 3, 5, 1, 0},
        {9,  4, 5, 2, 0},
        {13, 2, 7, 1, 1},
        {8,  5, 3, 3, 0},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int L = cases[c].L, C = cases[c].C, K = cases[c].K;
        const int dil = cases[c].dil, causal = cases[c].causal;
        unsigned seed = 1234 + (unsigned)c;

        struct tctx t;
        tctx_begin(&t);

        struct ggml_tensor * a   = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, L, C);  // [L, C]
        struct ggml_tensor * act = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, C, L);  // [C, L]
        struct ggml_tensor * w   = ggml_new_tensor_3d(t.ctx, GGML_TYPE_F32, K, 1, C);
        struct ggml_tensor * b   = ggml_new_tensor_1d(t.ctx, GGML_TYPE_F32, C);

        struct ggml_tensor * r0 = ggml_supertonic_depthwise_1d(t.ctx, a, w, b, dil);      // [L, C]
        struct ggml_tensor * r1 = ggml_supertonic_depthwise_1d_ct(t.ctx, act, w, b, dil); // [C, L]
        struct ggml_tensor * r2 = causal ? ggml_supertonic_depthwise_1d_causal_ct(t.ctx, act, w, b, dil) : NULL;

        CHECK((int)r0->ne[0] == L && (int)r0->ne[1] == C, "shape r0 (case %zu)", c);
        if (causal) CHECK((int)r2->ne[0] == C && (int)r2->ne[1] == L, "shape r2 (case %zu)", c);

        if (!tctx_alloc_graph(&t, r0, r1, r2)) { tctx_end(&t); continue; }

        const int n_a = L*C, n_w = K*C, n_b = C;
        float * pa  = (float *)malloc(n_a * sizeof(float));
        float * pat = (float *)malloc(n_a * sizeof(float));
        float * pw  = (float *)malloc(n_w * sizeof(float));
        float * pbb = (float *)malloc(n_b * sizeof(float));
        for (int ch = 0; ch < C; ch++)
            for (int tt = 0; tt < L; tt++) {
                const float v = frand(&seed);
                pa[tt + ch*L]  = v;   // (t, c) in [L, C]
                pat[ch + tt*C] = v;   // (c, t) in [C, L]
            }
        for (int i = 0; i < n_w; i++) pw[i] = frand(&seed);   // w[k + c*K]
        for (int i = 0; i < n_b; i++) pbb[i] = frand(&seed);

        tctx_upload(&t, a, pa);
        tctx_upload(&t, act, pat);
        tctx_upload(&t, w, pw);
        tctx_upload(&t, b, pbb);
        tctx_compute(&t);

        float * o0 = (float *)malloc(ggml_nbytes(r0));
        float * o1 = (float *)malloc(ggml_nbytes(r1));
        float * o2 = causal ? (float *)malloc(ggml_nbytes(r2)) : NULL;
        tctx_download(&t, r0, o0);
        tctx_download(&t, r1, o1);
        if (causal) tctx_download(&t, r2, o2);

        const int k_off_c = causal ? -(K - 1) : -(K / 2);   // for r2 (causal_ct)
        const int k_off_n = -(K / 2);                        // for r0/r1 (symmetric)
        int pass = 1;
        for (int ch = 0; ch < C && pass; ch++) {
            for (int tt = 0; tt < L && pass; tt++) {
                float sum_n = pbb[ch];
                float sum_c = pbb[ch];
                for (int k = 0; k < K; k++) {
                    int sn = tt + (k + k_off_n) * dil;
                    if (sn < 0) sn = 0; else if (sn >= L) sn = L - 1;
                    sum_n += pa[sn + ch*L] * pw[k + ch*K];
                    if (causal) {
                        int sc = tt + (k + k_off_c) * dil;
                        if (sc < 0) sc = 0; else if (sc >= L) sc = L - 1;
                        sum_c += pa[sc + ch*L] * pw[k + ch*K];
                    }
                }
                if (fabsf(o0[tt + ch*L] - sum_n) > 1e-3f) {
                    printf("    r0 mismatch case %d t%d c%d: %f vs %f\n", (int)c, tt, ch, o0[tt + ch*L], sum_n);
                    pass = 0;
                }
                if (fabsf(o1[ch + tt*C] - sum_n) > 1e-3f) {
                    printf("    r1 mismatch case %d t%d c%d: %f vs %f\n", (int)c, tt, ch, o1[ch + tt*C], sum_n);
                    pass = 0;
                }
                if (causal && fabsf(o2[ch + tt*C] - sum_c) > 1e-3f) {
                    printf("    r2 mismatch case %d t%d c%d: %f vs %f\n", (int)c, tt, ch, o2[ch + tt*C], sum_c);
                    pass = 0;
                }
            }
        }
        CHECK(pass, "depthwise_1d mismatch (case %zu L%d C%d K%d dil%d causal%d)",
              c, L, C, K, dil, causal);

        free(pa); free(pat); free(pw); free(pbb); free(o0); free(o1); free(o2);
        tctx_end(&t);
    }
    printf("  done (%d failures so far)\n", failures);
}

// ---------- 2. supertonic_layer_norm_channel ----------

static void test_supertonic_layer_norm_channel(void) {
    printf("[test] supertonic_layer_norm_channel / _ct\n");

    struct { int L, C; float eps; } cases[] = {
        {7, 4, 1e-5f},
        {16, 8, 1e-6f},
        {5, 3, 1e-5f},
        // Multi-simdgroup stress cases. The large L makes missing barriers in
        // the shared mean/variance reduction reliably observable on Metal.
        {4096, 64, 1e-5f},
        {4096, 256, 1e-5f},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int L = cases[c].L, C = cases[c].C;
        const float eps = cases[c].eps;
        unsigned seed = 777 + (unsigned)c;

        struct tctx t;
        tctx_begin(&t);

        struct ggml_tensor * a   = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, L, C);
        struct ggml_tensor * act = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, C, L);
        struct ggml_tensor * g   = ggml_new_tensor_1d(t.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * b   = ggml_new_tensor_1d(t.ctx, GGML_TYPE_F32, C);

        struct ggml_tensor * r0 = ggml_supertonic_layer_norm_channel(t.ctx, a, g, b, eps);
        struct ggml_tensor * r1 = ggml_supertonic_layer_norm_channel_ct(t.ctx, act, g, b, eps);

        if (!tctx_alloc_graph(&t, r0, r1, NULL)) { tctx_end(&t); continue; }

        const int n = L*C;
        float * pa  = (float *)malloc(n * sizeof(float));
        float * pat = (float *)malloc(n * sizeof(float));
        float * pg  = (float *)malloc(C * sizeof(float));
        float * pb  = (float *)malloc(C * sizeof(float));
        for (int ch = 0; ch < C; ch++)
            for (int tt = 0; tt < L; tt++) {
                const float v = frand(&seed) * 3.0f;
                pa[tt + ch*L]  = v;
                pat[ch + tt*C] = v;
            }
        for (int i = 0; i < C; i++) pg[i] = frand(&seed);
        for (int i = 0; i < C; i++) pb[i] = frand(&seed);

        tctx_upload(&t, a, pa);
        tctx_upload(&t, act, pat);
        tctx_upload(&t, g, pg);
        tctx_upload(&t, b, pb);
        tctx_compute(&t);

        float * o0 = (float *)malloc(ggml_nbytes(r0));
        float * o1 = (float *)malloc(ggml_nbytes(r1));
        tctx_download(&t, r0, o0);
        tctx_download(&t, r1, o1);

        int pass = 1;
        for (int tt = 0; tt < L && pass; tt++) {
            double mean = 0.0;
            for (int ch = 0; ch < C; ch++) mean += pa[tt + ch*L];
            mean /= C;
            double var = 0.0;
            for (int ch = 0; ch < C; ch++) {
                const double d = pa[tt + ch*L] - mean;
                var += d * d;
            }
            const float inv = (float)(1.0 / sqrt(var / C + eps));
            for (int ch = 0; ch < C; ch++) {
                const float want = (float)((pa[tt + ch*L] - mean)) * inv * pg[ch] + pb[ch];
                if (fabsf(o0[tt + ch*L] - want) > 1e-3f || fabsf(o1[ch + tt*C] - want) > 1e-3f) {
                    printf("    mismatch case %d t%d c%d: %f %f vs %f\n", (int)c, tt, ch,
                           o0[tt + ch*L], o1[ch + tt*C], want);
                    pass = 0;
                }
            }
        }
        CHECK(pass, "layer_norm_channel mismatch (case %zu L%d C%d)", c, L, C);

        free(pa); free(pat); free(pg); free(pb); free(o0); free(o1);
        tctx_end(&t);
    }
    printf("  done (%d failures so far)\n", failures);
}

// ---------- 3. supertonic_pw2_residual ----------

static void test_supertonic_pw2_residual(void) {
    printf("[test] supertonic_pw2_residual / _ct\n");

    struct { int L, C; } cases[] = { {9, 4}, {5, 7} };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int L = cases[c].L, C = cases[c].C;
        unsigned seed = 4242 + (unsigned)c;

        struct tctx t;
        tctx_begin(&t);

        struct ggml_tensor * x     = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, L, C);
        struct ggml_tensor * xt    = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, C, L);
        struct ggml_tensor * bias  = ggml_new_tensor_1d(t.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * gamma = ggml_new_tensor_1d(t.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * res   = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, L, C);
        struct ggml_tensor * rest  = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, C, L);

        struct ggml_tensor * r0 = ggml_supertonic_pw2_residual(t.ctx, x, bias, gamma, res);
        struct ggml_tensor * r1 = ggml_supertonic_pw2_residual_ct(t.ctx, xt, bias, gamma, rest);

        if (!tctx_alloc_graph(&t, r0, r1, NULL)) { tctx_end(&t); continue; }

        const int n = L*C;
        float * px    = (float *)malloc(n * sizeof(float));
        float * pxt   = (float *)malloc(n * sizeof(float));
        float * pr    = (float *)malloc(n * sizeof(float));
        float * prt   = (float *)malloc(n * sizeof(float));
        float * pb    = (float *)malloc(C * sizeof(float));
        float * pg    = (float *)malloc(C * sizeof(float));
        for (int ch = 0; ch < C; ch++)
            for (int tt = 0; tt < L; tt++) {
                const float vx = frand(&seed);
                const float vr = frand(&seed);
                px[tt + ch*L]   = vx;  pxt[ch + tt*C] = vx;
                pr[tt + ch*L]   = vr;  prt[ch + tt*C] = vr;
            }
        for (int i = 0; i < C; i++) pb[i] = frand(&seed);
        for (int i = 0; i < C; i++) pg[i] = frand(&seed);

        tctx_upload(&t, x, px);
        tctx_upload(&t, xt, pxt);
        tctx_upload(&t, res, pr);
        tctx_upload(&t, rest, prt);
        tctx_upload(&t, bias, pb);
        tctx_upload(&t, gamma, pg);
        tctx_compute(&t);

        float * o0 = (float *)malloc(ggml_nbytes(r0));
        float * o1 = (float *)malloc(ggml_nbytes(r1));
        tctx_download(&t, r0, o0);
        tctx_download(&t, r1, o1);

        int pass = 1;
        for (int tt = 0; tt < L && pass; tt++) {
            for (int ch = 0; ch < C && pass; ch++) {
                const float want = pr[tt + ch*L] + (px[tt + ch*L] + pb[ch]) * pg[ch];
                if (fabsf(o0[tt + ch*L] - want) > 1e-4f || fabsf(o1[ch + tt*C] - want) > 1e-4f) {
                    printf("    mismatch case %d t%d c%d: %f %f vs %f\n", (int)c, tt, ch,
                           o0[tt + ch*L], o1[ch + tt*C], want);
                    pass = 0;
                }
            }
        }
        CHECK(pass, "pw2_residual mismatch (case %zu L%d C%d)", c, L, C);

        free(px); free(pxt); free(pr); free(prt); free(pb); free(pg); free(o0); free(o1);
        tctx_end(&t);
    }
    printf("  done (%d failures so far)\n", failures);
}

// ---------- 4. supertonic_bias_gelu ----------

static void test_supertonic_bias_gelu(void) {
    printf("[test] supertonic_bias_gelu / _ct\n");

    struct { int L, C; } cases[] = { {9, 4}, {16, 3} };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int L = cases[c].L, C = cases[c].C;
        unsigned seed = 99 + (unsigned)c;

        struct tctx t;
        tctx_begin(&t);

        struct ggml_tensor * x    = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, L, C);
        struct ggml_tensor * xt   = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, C, L);
        struct ggml_tensor * bias = ggml_new_tensor_1d(t.ctx, GGML_TYPE_F32, C);

        struct ggml_tensor * r0 = ggml_supertonic_bias_gelu(t.ctx, x, bias);
        struct ggml_tensor * r1 = ggml_supertonic_bias_gelu_ct(t.ctx, xt, bias);

        if (!tctx_alloc_graph(&t, r0, r1, NULL)) { tctx_end(&t); continue; }

        const int n = L*C;
        float * px  = (float *)malloc(n * sizeof(float));
        float * pxt = (float *)malloc(n * sizeof(float));
        float * pb  = (float *)malloc(C * sizeof(float));
        for (int ch = 0; ch < C; ch++)
            for (int tt = 0; tt < L; tt++) {
                const float v = frand(&seed) * 4.0f;
                px[tt + ch*L]  = v;
                pxt[ch + tt*C] = v;
            }
        for (int i = 0; i < C; i++) pb[i] = frand(&seed);

        tctx_upload(&t, x, px);
        tctx_upload(&t, xt, pxt);
        tctx_upload(&t, bias, pb);
        tctx_compute(&t);

        float * o0 = (float *)malloc(ggml_nbytes(r0));
        float * o1 = (float *)malloc(ggml_nbytes(r1));
        tctx_download(&t, r0, o0);
        tctx_download(&t, r1, o1);

        int pass = 1;
        for (int tt = 0; tt < L && pass; tt++) {
            for (int ch = 0; ch < C && pass; ch++) {
                const float want = gelu_erf_f(px[tt + ch*L] + pb[ch]);
                if (fabsf(o0[tt + ch*L] - want) > 1e-4f || fabsf(o1[ch + tt*C] - want) > 1e-4f) {
                    printf("    mismatch case %d t%d c%d: %f %f vs %f\n", (int)c, tt, ch,
                           o0[tt + ch*L], o1[ch + tt*C], want);
                    pass = 0;
                }
            }
        }
        CHECK(pass, "bias_gelu mismatch (case %zu L%d C%d)", c, L, C);

        free(px); free(pxt); free(pb); free(o0); free(o1);
        tctx_end(&t);
    }
    printf("  done (%d failures so far)\n", failures);
}

// ---------- 5. supertonic_edge_pad_1d ----------

static void test_supertonic_edge_pad_1d(void) {
    printf("[test] supertonic_edge_pad_1d / _ct\n");

    struct { int L, C, pl, prr; } cases[] = {
        {8, 3, 2, 2},
        {8, 3, 3, 0},
        {5, 4, 0, 2},
        {4, 2, 5, 5},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int L = cases[c].L, C = cases[c].C, pl = cases[c].pl, prr = cases[c].prr;
        const int LO = L + pl + prr;
        unsigned seed = 555 + (unsigned)c;

        struct tctx t;
        tctx_begin(&t);

        struct ggml_tensor * x  = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, L, C);
        struct ggml_tensor * xt = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, C, L);

        struct ggml_tensor * r0 = ggml_supertonic_edge_pad_1d(t.ctx, x, pl, prr);   // [LO, C]
        struct ggml_tensor * r1 = ggml_supertonic_edge_pad_1d_ct(t.ctx, xt, pl, prr); // [C, LO]

        CHECK((int)r0->ne[0] == LO && (int)r0->ne[1] == C, "shape r0 (case %zu)", c);
        CHECK((int)r1->ne[0] == C && (int)r1->ne[1] == LO, "shape r1 (case %zu)", c);

        if (!tctx_alloc_graph(&t, r0, r1, NULL)) { tctx_end(&t); continue; }

        const int n = L*C;
        float * px  = (float *)malloc(n * sizeof(float));
        float * pxt = (float *)malloc(n * sizeof(float));
        for (int ch = 0; ch < C; ch++)
            for (int tt = 0; tt < L; tt++) {
                const float v = frand(&seed);
                px[tt + ch*L]  = v;
                pxt[ch + tt*C] = v;
            }

        tctx_upload(&t, x, px);
        tctx_upload(&t, xt, pxt);
        tctx_compute(&t);

        float * o0 = (float *)malloc(LO*C * sizeof(float));
        float * o1 = (float *)malloc(LO*C * sizeof(float));
        tctx_download(&t, r0, o0);
        tctx_download(&t, r1, o1);

        int pass = 1;
        for (int ch = 0; ch < C && pass; ch++) {
            for (int tt = 0; tt < LO && pass; tt++) {
                int s = tt - pl;
                if (s < 0) s = 0;
                if (s >= L) s = L - 1;
                const float want = px[s + ch*L];
                if (fabsf(o0[tt + ch*LO] - want) > 1e-6f || fabsf(o1[ch + tt*C] - want) > 1e-6f) {
                    printf("    mismatch case %d t%d c%d: %f %f vs %f\n", (int)c, tt, ch,
                           o0[tt + ch*LO], o1[ch + tt*C], want);
                    pass = 0;
                }
            }
        }
        CHECK(pass, "edge_pad_1d mismatch (case %zu L%d C%d pl%d pr%d)", c, L, C, pl, prr);

        free(px); free(pxt); free(o0); free(o1);
        tctx_end(&t);
    }
    printf("  done (%d failures so far)\n", failures);
}

// ---------- 6. gru ----------

static void test_gru(void) {
    printf("[test] gru (forward + reverse)\n");

    struct { int H, B, L; int rev; } cases[] = {
        {4, 1, 6, 0},
        {4, 3, 6, 0},
        {8, 2, 5, 1},
        {16, 2, 4, 0},
        {2, 4, 7, 1},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int H = cases[c].H, B = cases[c].B, L = cases[c].L, rev = cases[c].rev;
        unsigned seed = 31337 + (unsigned)c;

        struct tctx t;
        tctx_begin(&t);

        struct ggml_tensor * whh = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, H, 3*H);  // [H, 3H]
        struct ggml_tensor * gi  = ggml_new_tensor_3d(t.ctx, GGML_TYPE_F32, 3*H, B, L);
        struct ggml_tensor * bhh = ggml_new_tensor_1d(t.ctx, GGML_TYPE_F32, 3*H);

        struct ggml_tensor * r = ggml_gru(t.ctx, whh, gi, bhh, rev != 0);

        CHECK((int)r->ne[0] == H && (int)r->ne[1] == B && (int)r->ne[2] == L, "shape (case %zu)", c);

        if (!tctx_alloc_graph(&t, r, NULL, NULL)) { tctx_end(&t); continue; }

        const int n_whh = H*3*H, n_gi = 3*H*B*L, n_bhh = 3*H;
        float * p_whh = (float *)malloc(n_whh * sizeof(float));
        float * p_gi  = (float *)malloc(n_gi * sizeof(float));
        float * p_bhh = (float *)malloc(n_bhh * sizeof(float));
        for (int i = 0; i < n_whh; i++) p_whh[i] = frand(&seed);
        for (int i = 0; i < n_gi; i++) p_gi[i] = frand(&seed);
        for (int i = 0; i < n_bhh; i++) p_bhh[i] = frand(&seed);

        tctx_upload(&t, whh, p_whh);
        tctx_upload(&t, gi, p_gi);
        tctx_upload(&t, bhh, p_bhh);
        tctx_compute(&t);

        float * out = (float *)malloc(ggml_nbytes(r));
        tctx_download(&t, r, out);

        // reference: serial over time, parallel over batch.  PyTorch GRU semantics:
        //   gh[g] = sum_k whh[k + g*H] * h[k] + bhh[g]   (whh [H, 3H] col-major)
        //   r = sigmoid(gi[r] + gh[r]); z = sigmoid(gi[z] + gh[z]); n = tanh(gi[n] + r*gh[n])
        //   h' = n + z*(h - n)
        float * h = (float *)calloc(H, sizeof(float));
        float * gh = (float *)calloc(3*H, sizeof(float));
        int pass = 1;
        for (int b = 0; b < B && pass; b++) {
            memset(h, 0, H * sizeof(float));
            for (int s = 0; s < L && pass; s++) {
                const int tt = rev ? (L - 1 - s) : s;
                const float * gip = p_gi + ((size_t)3*H*b + (size_t)3*H*B*tt);
                for (int g = 0; g < 3*H; g++) {
                    float acc = 0.0f;
                    for (int k = 0; k < H; k++) acc += p_whh[k + (size_t)H*g] * h[k];
                    gh[g] = acc + p_bhh[g];
                }
                for (int j = 0; j < H; j++) {
                    const float rr = 1.0f/(1.0f + expf(-(gip[j] + gh[j])));
                    const float zz = 1.0f/(1.0f + expf(-(gip[H + j] + gh[H + j])));
                    const float nn = tanhf(gip[2*H + j] + rr*gh[2*H + j]);
                    h[j] = nn + zz*(h[j] - nn);
                    const float got = out[(size_t)H*b + (size_t)H*B*tt + j];
                    if (fabsf(got - h[j]) > 2e-3f) {
                        printf("    mismatch case %d b%d t%d j%d: %f vs %f\n", (int)c, b, tt, j, got, h[j]);
                        pass = 0;
                    }
                }
            }
        }
        CHECK(pass, "gru mismatch (case %zu H%d B%d L%d rev%d)", c, H, B, L, rev);
        free(h); free(gh);

        free(p_whh); free(p_gi); free(p_bhh); free(out);
        tctx_end(&t);
    }
    printf("  done (%d failures so far)\n", failures);
}

// ---------- 7. zero_upsample ----------

static void test_zero_upsample(void) {
    printf("[test] zero_upsample\n");

    struct { int F, R, s; } cases[] = {
        {5, 1, 2},
        {5, 3, 3},
        {1, 2, 4},
        {9, 2, 2},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int F = cases[c].F, R = cases[c].R, s = cases[c].s;
        const int Fu = (F - 1)*s + 1;
        unsigned seed = 2024 + (unsigned)c;

        struct tctx t;
        tctx_begin(&t);

        struct ggml_tensor * a = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, F, R);
        struct ggml_tensor * r = ggml_zero_upsample(t.ctx, a, s);

        CHECK((int)r->ne[0] == Fu && (int)r->ne[1] == R, "shape (case %zu)", c);

        if (!tctx_alloc_graph(&t, r, NULL, NULL)) { tctx_end(&t); continue; }

        const int n = F*R;
        float * pa = (float *)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) pa[i] = frand(&seed);
        tctx_upload(&t, a, pa);
        tctx_compute(&t);

        float * out = (float *)malloc(ggml_nbytes(r));
        tctx_download(&t, r, out);

        int pass = 1;
        for (int rr = 0; rr < R && pass; rr++) {
            for (int f = 0; f < Fu && pass; f++) {
                const float want = (f % s == 0) ? pa[f/s + rr*F] : 0.0f;
                if (fabsf(out[f + rr*Fu] - want) > 1e-6f) {
                    printf("    mismatch case %d r%d f%d: %f vs %f\n", (int)c, rr, f, out[f + rr*Fu], want);
                    pass = 0;
                }
            }
        }
        CHECK(pass, "zero_upsample mismatch (case %zu F%d R%d s%d)", c, F, R, s);

        free(pa); free(out);
        tctx_end(&t);
    }
    printf("  done (%d failures so far)\n", failures);
}

// ---------- 8. channel_shuffle ----------

static void test_channel_shuffle(void) {
    printf("[test] channel_shuffle\n");

    struct { int F, T, C, Bc, G; } cases[] = {
        {3, 4, 6, 1, 2},
        {3, 4, 6, 2, 3},
        {2, 2, 4, 1, 4},
        {5, 2, 8, 2, 2},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int F = cases[c].F, T = cases[c].T, C = cases[c].C, Bc = cases[c].Bc, G = cases[c].G;
        const int FT = F*T;
        unsigned seed = 606 + (unsigned)c;

        struct tctx t;
        tctx_begin(&t);

        struct ggml_tensor * a = ggml_new_tensor_4d(t.ctx, GGML_TYPE_F32, F, T, C, Bc);
        struct ggml_tensor * r = ggml_channel_shuffle(t.ctx, a, G);

        if (!tctx_alloc_graph(&t, r, NULL, NULL)) { tctx_end(&t); continue; }

        const int n = FT*C*Bc;
        float * pa = (float *)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) pa[i] = frand(&seed);
        tctx_upload(&t, a, pa);
        tctx_compute(&t);

        float * out = (float *)malloc(ggml_nbytes(r));
        tctx_download(&t, r, out);

        const int cg = C / G;
        int pass = 1;
        for (int cp = 0; cp < C && pass; cp++) {
            const int in_c = (cp % G) * cg + cp / G;
            for (int b = 0; b < Bc && pass; b++) {
                for (int i = 0; i < FT && pass; i++) {
                    const float want = pa[(in_c + C*b)*FT + i];
                    const float got = out[(cp + C*b)*FT + i];
                    if (fabsf(got - want) > 1e-6f) {
                        printf("    mismatch case %d cp%d b%d i%d: %f vs %f\n", (int)c, cp, b, i, got, want);
                        pass = 0;
                    }
                }
            }
        }
        CHECK(pass, "channel_shuffle mismatch (case %zu C%d G%d)", c, C, G);

        free(pa); free(out);
        tctx_end(&t);
    }
    printf("  done (%d failures so far)\n", failures);
}

// ---------- 9. affine_prelu ----------

static void test_affine_prelu(void) {
    printf("[test] affine_prelu\n");

    struct { int F, T, C, Bc; } cases[] = {
        {4, 5, 3, 1},
        {4, 5, 3, 2},
        {2, 3, 5, 1},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int F = cases[c].F, T = cases[c].T, C = cases[c].C, Bc = cases[c].Bc;
        const int FT = F*T;
        unsigned seed = 808 + (unsigned)c;

        struct tctx t;
        tctx_begin(&t);

        struct ggml_tensor * x     = ggml_new_tensor_4d(t.ctx, GGML_TYPE_F32, F, T, C, Bc);
        struct ggml_tensor * aw    = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, F, C);
        struct ggml_tensor * ab    = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, F, C);
        struct ggml_tensor * slope = ggml_new_tensor_1d(t.ctx, GGML_TYPE_F32, C);

        struct ggml_tensor * r = ggml_affine_prelu(t.ctx, x, aw, ab, slope);

        if (!tctx_alloc_graph(&t, r, NULL, NULL)) { tctx_end(&t); continue; }

        const int n = FT*C*Bc;
        float * px  = (float *)malloc(n * sizeof(float));
        float * paw = (float *)malloc(F*C * sizeof(float));
        float * pab = (float *)malloc(F*C * sizeof(float));
        float * psl = (float *)malloc(C * sizeof(float));
        for (int i = 0; i < n; i++) px[i] = frand(&seed) * 2.0f;
        for (int i = 0; i < F*C; i++) paw[i] = frand(&seed);
        for (int i = 0; i < F*C; i++) pab[i] = frand(&seed);
        for (int i = 0; i < C; i++) psl[i] = frand(&seed);

        tctx_upload(&t, x, px);
        tctx_upload(&t, aw, paw);
        tctx_upload(&t, ab, pab);
        tctx_upload(&t, slope, psl);
        tctx_compute(&t);

        float * out = (float *)malloc(ggml_nbytes(r));
        tctx_download(&t, r, out);

        int pass = 1;
        for (int p = 0; p < C*Bc && pass; p++) {
            const int ch = p % C;
            for (int i = 0; i < FT && pass; i++) {
                const float xv = px[p*FT + i];
                const float rv = xv > 0.0f ? xv : 0.0f;
                const float want = xv * paw[ch*F + (i % F)] + pab[ch*F + (i % F)]
                                 + rv + psl[ch] * (xv - rv);
                if (fabsf(out[p*FT + i] - want) > 1e-4f) {
                    printf("    mismatch case %d p%d i%d: %f vs %f\n", (int)c, p, i, out[p*FT + i], want);
                    pass = 0;
                }
            }
        }
        CHECK(pass, "affine_prelu mismatch (case %zu C%d Bc%d)", c, C, Bc);

        free(px); free(paw); free(pab); free(psl); free(out);
        tctx_end(&t);
    }
    printf("  done (%d failures so far)\n", failures);
}

// ---------- 10. snake ----------

static void test_snake(void) {
    printf("[test] snake\n");

    struct { int T, C; } cases[] = { {16, 4}, {5, 8} };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int T = cases[c].T, C = cases[c].C;
        unsigned seed = 909 + (unsigned)c;

        struct tctx t;
        tctx_begin(&t);

        struct ggml_tensor * x  = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, T, C);
        struct ggml_tensor * a  = ggml_new_tensor_1d(t.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * ib = ggml_new_tensor_1d(t.ctx, GGML_TYPE_F32, C);

        struct ggml_tensor * r = ggml_snake(t.ctx, x, a, ib);

        if (!tctx_alloc_graph(&t, r, NULL, NULL)) { tctx_end(&t); continue; }

        const int n = T*C;
        float * px  = (float *)malloc(n * sizeof(float));
        float * pa  = (float *)malloc(C * sizeof(float));
        float * pib = (float *)malloc(C * sizeof(float));
        for (int i = 0; i < n; i++) px[i] = frand(&seed) * 3.0f;
        for (int i = 0; i < C; i++) pa[i] = frand(&seed);
        for (int i = 0; i < C; i++) pib[i] = frand(&seed);

        tctx_upload(&t, x, px);
        tctx_upload(&t, a, pa);
        tctx_upload(&t, ib, pib);
        tctx_compute(&t);

        float * out = (float *)malloc(ggml_nbytes(r));
        tctx_download(&t, r, out);

        int pass = 1;
        for (int ch = 0; ch < C && pass; ch++) {
            for (int tt = 0; tt < T && pass; tt++) {
                const float xi = px[tt + ch*T];
                const float si = sinf(pa[ch] * xi);
                const float want = xi + si*si*pib[ch];
                if (fabsf(out[tt + ch*T] - want) > 1e-4f) {
                    printf("    mismatch case %d t%d c%d: %f vs %f\n", (int)c, tt, ch, out[tt + ch*T], want);
                    pass = 0;
                }
            }
        }
        CHECK(pass, "snake mismatch (case %zu T%d C%d)", c, T, C);

        free(px); free(pa); free(pib); free(out);
        tctx_end(&t);
    }
    printf("  done (%d failures so far)\n", failures);
}

#ifdef USE_METAL
static void test_metal_support_gates(void) {
    if (strcmp(g_backend, "metal") != 0) {
        return;
    }

    printf("[test] metal supports_op gates / fallback envelope\n");

    struct tctx t;
    tctx_begin(&t);

    struct ggml_tensor * x_tc = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, 8, 4);
    struct ggml_tensor * w    = ggml_new_tensor_3d(t.ctx, GGML_TYPE_F32, 3, 1, 4);
    struct ggml_tensor * v4a  = ggml_new_tensor_1d(t.ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * v4b  = ggml_new_tensor_1d(t.ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * res  = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, 8, 4);

    struct ggml_tensor * depthwise = ggml_supertonic_depthwise_1d(t.ctx, x_tc, w, v4a, 1);
    CHECK(ggml_backend_supports_op(t.backend, depthwise), "metal depthwise supported envelope rejected");
    ((int32_t *) depthwise->op_params)[0] = 9;
    CHECK(!ggml_backend_supports_op(t.backend, depthwise), "metal depthwise K=9 must fall back");

    struct ggml_tensor * layer_norm = ggml_supertonic_layer_norm_channel(t.ctx, x_tc, v4a, v4b, 1e-5f);
    CHECK(ggml_backend_supports_op(t.backend, layer_norm), "metal layer_norm supported envelope rejected");
    ((int32_t *) layer_norm->op_params)[1] = 2;
    CHECK(!ggml_backend_supports_op(t.backend, layer_norm), "metal layer_norm invalid layout must fall back");

    struct ggml_tensor * pw2 = ggml_supertonic_pw2_residual(t.ctx, x_tc, v4a, v4b, res);
    CHECK(ggml_backend_supports_op(t.backend, pw2), "metal pw2 supported envelope rejected");
    ((int32_t *) pw2->op_params)[0] = 2;
    CHECK(!ggml_backend_supports_op(t.backend, pw2), "metal pw2 invalid layout must fall back");

    struct ggml_tensor * bias_gelu = ggml_supertonic_bias_gelu(t.ctx, x_tc, v4a);
    CHECK(ggml_backend_supports_op(t.backend, bias_gelu), "metal bias_gelu supported envelope rejected");
    ((int32_t *) bias_gelu->op_params)[0] = 2;
    CHECK(!ggml_backend_supports_op(t.backend, bias_gelu), "metal bias_gelu invalid layout must fall back");

    struct ggml_tensor * edge_pad = ggml_supertonic_edge_pad_1d(t.ctx, x_tc, 1, 2);
    CHECK(ggml_backend_supports_op(t.backend, edge_pad), "metal edge_pad supported envelope rejected");
    ((int32_t *) edge_pad->op_params)[2] = 2;
    CHECK(!ggml_backend_supports_op(t.backend, edge_pad), "metal edge_pad invalid layout must fall back");

    struct ggml_tensor * snake = ggml_snake(t.ctx, x_tc, v4a, v4b);
    CHECK(ggml_backend_supports_op(t.backend, snake), "metal snake supported envelope rejected");
    const enum ggml_type saved_type = v4b->type;
    v4b->type = GGML_TYPE_F16;
    CHECK(!ggml_backend_supports_op(t.backend, snake), "metal snake mixed types must fall back");
    v4b->type = saved_type;

    struct ggml_tensor * whh = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, 4, 12);
    struct ggml_tensor * gi  = ggml_new_tensor_3d(t.ctx, GGML_TYPE_F32, 12, 1, 3);
    struct ggml_tensor * bhh = ggml_new_tensor_1d(t.ctx, GGML_TYPE_F32, 12);
    CHECK(!ggml_backend_supports_op(t.backend, ggml_gru(t.ctx, whh, gi, bhh, false)),
          "metal GRU must fall back without a kernel");

    CHECK(!ggml_backend_supports_op(t.backend, ggml_zero_upsample(t.ctx, x_tc, 2)),
          "metal zero_upsample must fall back without a kernel");

    struct ggml_tensor * x4d = ggml_new_tensor_4d(t.ctx, GGML_TYPE_F32, 2, 3, 4, 1);
    CHECK(!ggml_backend_supports_op(t.backend, ggml_channel_shuffle(t.ctx, x4d, 2)),
          "metal channel_shuffle must fall back without a kernel");

    struct ggml_tensor * aw    = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, 2, 4);
    struct ggml_tensor * ab    = ggml_new_tensor_2d(t.ctx, GGML_TYPE_F32, 2, 4);
    struct ggml_tensor * slope = ggml_new_tensor_1d(t.ctx, GGML_TYPE_F32, 4);
    CHECK(!ggml_backend_supports_op(t.backend, ggml_affine_prelu(t.ctx, x4d, aw, ab, slope)),
          "metal affine_prelu must fall back without a kernel");

    tctx_end(&t);
    printf("  done (%d failures so far)\n", failures);
}
#endif

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc > 1) {
        g_backend = argv[1];
    }
    printf("== qvac-ops smoke tests (backend: %s) ==\n", g_backend);

    test_supertonic_depthwise_1d();
    test_supertonic_layer_norm_channel();
    test_supertonic_pw2_residual();
    test_supertonic_bias_gelu();
    test_supertonic_edge_pad_1d();
    test_gru();
    test_zero_upsample();
    test_channel_shuffle();
    test_affine_prelu();
    test_snake();
#ifdef USE_METAL
    test_metal_support_gates();
#endif

    if (failures == 0) {
        printf("\nALL PASSED\n");
        return 0;
    }
    printf("\n%d FAILURES\n", failures);
    return 1;
}
