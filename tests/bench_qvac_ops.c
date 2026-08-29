// Benchmark: qvac-ops fused kernels vs equivalent composed sub-graphs.
//
// Usage: bench_qvac_ops [cpu|vulkan|metal] [threads]
//
// Measures end-to-end graph_compute time (median over repeats):
//   1. snake            vs  mul+sin+sqr+mul+add chain (the 5-op subgraph the
//                           upstream Vulkan backend has a fusion hook for)
//   2. bias_gelu        vs  add + gelu_erf
//   3. pw2_residual     vs  add + mul + add
//   4. affine_prelu     vs  prelu-composed chain (max/min + mul + add)
//   5. channel_shuffle  vs  reshape+permute+cont chain
//   6. zero_upsample    vs  im2col-free naive: repeat + diag-mask style zero
//                           fill (composed: concat with zeros via upscale)
//   7. supertonic_depthwise_1d vs pad + conv_1d_dw + bias add
//   8. supertonic_edge_pad_1d  vs concat of edge tiles
//   9. gru              (absolute; no native equivalent - RNN gap in ggml)
//  10. supertonic_layer_norm_channel vs permute+cont+norm+mul+add+permute+cont
//
// Reports median over N repeats, ms.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef _WIN32
#include <windows.h>
static double now_s(void) {
    static LARGE_INTEGER freq = {0}, base = {0};
    if (freq.QuadPart == 0) { QueryPerformanceFrequency(&freq); QueryPerformanceCounter(&base); return 0; }
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    return (double)(now.QuadPart - base.QuadPart) / (double)freq.QuadPart;
}
#else
#include <time.h>
static double now_s(void) {
    static struct timespec ts_base = {0,0};
    if (ts_base.tv_sec == 0) { clock_gettime(CLOCK_MONOTONIC, &ts_base); return 0; }
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec - ts_base.tv_sec) + (ts.tv_nsec - ts_base.tv_nsec) * 1e-9;
}
#endif

#ifdef USE_VULKAN
ggml_backend_t             ggml_backend_vk_init(size_t dev_num);
ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
#endif
#ifdef USE_METAL
ggml_backend_t ggml_backend_metal_init(void);
#endif

static const char * g_backend = "cpu";
static int g_threads = 8;
static int g_warmup  = 3;
static int g_repeats = 20;

// ---- backend helpers (galloc + direct graph_compute, bypass sched) ----

struct bench {
    struct ggml_context * ctx;
    ggml_backend_t backend;
    ggml_backend_buffer_type_t buft;
    ggml_gallocr_t galloc;
    struct ggml_cgraph * gf;
};

static void bench_begin(struct bench * b) {
    size_t buf_size = 256u*1024u*1024u;
    struct ggml_init_params ip = { buf_size, NULL, /* no_alloc */ true };
    b->ctx = ggml_init(ip);
    b->backend = NULL; b->buft = NULL; b->galloc = NULL; b->gf = NULL;
#ifdef USE_VULKAN
    if (strcmp(g_backend, "vulkan") == 0) {
        b->backend = ggml_backend_vk_init(0);
        b->buft = ggml_backend_vk_buffer_type(0);
        return;
    }
#endif
#ifdef USE_METAL
    if (strcmp(g_backend, "metal") == 0) {
        b->backend = ggml_backend_metal_init();
        GGML_ASSERT(b->backend && "USE_METAL build but Metal initialization failed");
        b->buft = ggml_backend_get_default_buffer_type(b->backend);
        return;
    }
#endif
    b->backend = ggml_backend_cpu_init();
    b->buft = ggml_backend_cpu_buffer_type();
    ggml_backend_cpu_set_n_threads(b->backend, g_threads);
}

static bool bench_graph(struct bench * b, struct ggml_tensor * root) {
    b->gf = ggml_new_graph(b->ctx);
    ggml_build_forward_expand(b->gf, root);
    for (int i = 0; i < ggml_graph_n_nodes(b->gf); i++) {
        if (!ggml_backend_supports_op(b->backend, ggml_graph_node(b->gf, i))) {
            return false;
        }
    }
    b->galloc = ggml_gallocr_new(b->buft);
    if (!ggml_gallocr_alloc_graph(b->galloc, b->gf)) {
        printf("  galloc failed\n");
        return false;
    }
    return true;
}

static void bench_upload(struct bench * b, struct ggml_tensor * t, const void * host) {
    ggml_backend_tensor_set(t, host, 0, ggml_nbytes(t));
}

static double bench_time_median(struct bench * b) {
    double samples[64];
    int n = g_repeats < 64 ? g_repeats : 64;
    for (int i = 0; i < g_warmup; i++) ggml_backend_graph_compute(b->backend, b->gf);
    for (int i = 0; i < n; i++) {
        double t0 = now_s();
        ggml_backend_graph_compute(b->backend, b->gf);
        double t1 = now_s();
        samples[i] = (t1 - t0) * 1000.0;
    }
    // median
    for (int i = 1; i < n; i++) {
        double key = samples[i]; int j = i - 1;
        while (j >= 0 && samples[j] > key) { samples[j+1] = samples[j]; j--; }
        samples[j+1] = key;
    }
    return samples[n/2];
}

static void bench_end(struct bench * b) {
    if (b->galloc) ggml_gallocr_free(b->galloc);
    if (b->backend) ggml_backend_free(b->backend);
    ggml_free(b->ctx);
}

static float * fill_rand(int n, unsigned seed) {
    float * p = (float *)malloc((size_t)n * sizeof(float));
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        p[i] = (float)((s >> 16) % 2000) * 0.001f - 1.0f;
    }
    return p;
}

// measure one prebuilt graph; -1 when unsupported
static double time_variant(struct bench * b, struct ggml_tensor * root,
                           struct ggml_tensor ** inputs, float ** data, int n_in) {
    if (!ggml_backend_supports_op(b->backend, root)) {
        bench_end(b);
        return -1;
    }
    if (!bench_graph(b, root)) { bench_end(b); return -1; }
    for (int i = 0; i < n_in; i++) bench_upload(b, inputs[i], data[i]);
    double t = bench_time_median(b);
    bench_end(b);
    return t;
}

static void report(const char * name, double fused, double composed) {
    if (fused < 0 && composed < 0) {
        printf("  %-34s  SKIP (both unsupported)\n", name);
    } else if (composed < 0) {
        printf("  %-34s  fused %8.4f ms   (composed unsupported)\n", name, fused);
    } else if (fused < 0) {
        printf("  %-34s  composed %8.4f ms  (fused unsupported)\n", name, composed);
    } else {
        printf("  %-34s  fused %8.4f ms  composed %8.4f ms  speedup %.2fx\n",
               name, fused, composed, composed / fused);
    }
}

// ============ 1. snake vs 5-op chain ============

static void bench_snake(void) {
    printf("\n--- snake vs mul+sin+sqr+mul+add (%s) ---\n", g_backend);
    printf("  %8s %6s | %11s %11s %8s\n", "T", "C", "fused ms", "chain ms", "speedup");

    struct { int T, C; } cases[] = {
        {2048, 64}, {8192, 128}, {32768, 32},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int T = cases[c].T, C = cases[c].C;
        float * px  = fill_rand(T*C, 11 + (unsigned)c);
        float * pa  = fill_rand(C, 5 + (unsigned)c);
        float * pib = fill_rand(C, 9 + (unsigned)c);

        // fused
        struct bench b; bench_begin(&b);
        struct ggml_tensor * x  = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, T, C);
        struct ggml_tensor * a  = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * ib = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * rf = ggml_snake(b.ctx, x, a, ib);
        double tf = time_variant(&b, rf, (struct ggml_tensor *[]){x, a, ib},
                                 (float *[]){px, pa, pib}, 3);

        // composed chain: y = x + inv_b * sin(a*x)^2
        //   m1 = mul(x, a_broadcast) ; s = sin(m1) ; sq = sqr(s)
        //   m2 = mul(sq, inv_b_bcast); y = add(x, m2)
        bench_begin(&b);
        x  = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, T, C);
        a  = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        ib = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * a_b  = ggml_repeat(b.ctx, a, x);
        struct ggml_tensor * m1   = ggml_mul(b.ctx, x, a_b);
        struct ggml_tensor * sn   = ggml_sin(b.ctx, m1);
        struct ggml_tensor * sq   = ggml_sqr(b.ctx, sn);
        struct ggml_tensor * ib_b = ggml_repeat(b.ctx, ib, x);
        struct ggml_tensor * m2   = ggml_mul(b.ctx, sq, ib_b);
        struct ggml_tensor * rc   = ggml_add(b.ctx, x, m2);
        double tc = time_variant(&b, rc, (struct ggml_tensor *[]){x, a, ib},
                                 (float *[]){px, pa, pib}, 3);

        char name[64]; snprintf(name, sizeof(name), "snake %dx%d", T, C);
        report(name, tf, tc);
        free(px); free(pa); free(pib);
    }
}

// ============ 2. bias_gelu vs add+gelu ============

static void bench_bias_gelu(void) {
    printf("\n--- supertonic_bias_gelu vs add+gelu_erf (%s) ---\n", g_backend);
    printf("  %8s %6s | %11s %11s %8s\n", "L", "C", "fused ms", "chain ms", "speedup");

    struct { int L, C; } cases[] = {
        {1024, 256}, {4096, 512}, {1024, 1024},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int L = cases[c].L, C = cases[c].C;
        float * px = fill_rand(L*C, 21 + (unsigned)c);
        float * pb = fill_rand(C, 23 + (unsigned)c);

        struct bench b; bench_begin(&b);
        struct ggml_tensor * x = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, L, C);
        struct ggml_tensor * bi = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * rf = ggml_supertonic_bias_gelu(b.ctx, x, bi);
        double tf = time_variant(&b, rf, (struct ggml_tensor *[]){x, bi},
                                 (float *[]){px, pb}, 2);

        bench_begin(&b);
        x  = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, L, C);
        bi = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * bib = ggml_repeat(b.ctx, bi, x);
        struct ggml_tensor * ad  = ggml_add(b.ctx, x, bib);
        struct ggml_tensor * rc  = ggml_gelu_erf(b.ctx, ad);
        double tc = time_variant(&b, rc, (struct ggml_tensor *[]){x, bi},
                                 (float *[]){px, pb}, 2);

        char name[64]; snprintf(name, sizeof(name), "bias_gelu %dx%d", L, C);
        report(name, tf, tc);
        free(px); free(pb);
    }
}

// ============ 3. pw2_residual vs add+mul+add ============

static void bench_pw2_residual(void) {
    printf("\n--- supertonic_pw2_residual vs add+mul+add (%s) ---\n", g_backend);
    printf("  %8s %6s | %11s %11s %8s\n", "L", "C", "fused ms", "chain ms", "speedup");

    struct { int L, C; } cases[] = {
        {1024, 256}, {4096, 512},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int L = cases[c].L, C = cases[c].C;
        float * px = fill_rand(L*C, 31 + (unsigned)c);
        float * pr = fill_rand(L*C, 33 + (unsigned)c);
        float * pb = fill_rand(C, 35 + (unsigned)c);
        float * pg = fill_rand(C, 37 + (unsigned)c);

        struct bench b; bench_begin(&b);
        struct ggml_tensor * x  = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, L, C);
        struct ggml_tensor * rs = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, L, C);
        struct ggml_tensor * bi = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * ga = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * rf = ggml_supertonic_pw2_residual(b.ctx, x, bi, ga, rs);
        double tf = time_variant(&b, rf, (struct ggml_tensor *[]){x, rs, bi, ga},
                                 (float *[]){px, pr, pb, pg}, 4);

        bench_begin(&b);
        x  = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, L, C);
        rs = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, L, C);
        bi = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        ga = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * bib = ggml_repeat(b.ctx, bi, x);
        struct ggml_tensor * ad1 = ggml_add(b.ctx, x, bib);
        struct ggml_tensor * gab = ggml_repeat(b.ctx, ga, x);
        struct ggml_tensor * mu  = ggml_mul(b.ctx, ad1, gab);
        struct ggml_tensor * rc  = ggml_add(b.ctx, rs, mu);
        double tc = time_variant(&b, rc, (struct ggml_tensor *[]){x, rs, bi, ga},
                                 (float *[]){px, pr, pb, pg}, 4);

        char name[64]; snprintf(name, sizeof(name), "pw2_residual %dx%d", L, C);
        report(name, tf, tc);
        free(px); free(pr); free(pb); free(pg);
    }
}

// ============ 4. affine_prelu vs composed ============

static void bench_affine_prelu(void) {
    printf("\n--- affine_prelu vs affine+prelu chain (%s) ---\n", g_backend);
    printf("  %8s %6s %4s | %11s %11s %8s\n", "F", "T", "C", "fused ms", "chain ms", "speedup");

    struct { int F, T, C; } cases[] = {
        {64, 256, 32}, {128, 512, 64},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int F = cases[c].F, T = cases[c].T, C = cases[c].C;
        const int FT = F * T;
        float * px  = fill_rand(FT*C, 41 + (unsigned)c);
        float * paw = fill_rand(F*C, 43 + (unsigned)c);
        float * pab = fill_rand(F*C, 45 + (unsigned)c);
        float * psl = fill_rand(C, 47 + (unsigned)c);

        struct bench b; bench_begin(&b);
        struct ggml_tensor * x  = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F32, F, T, C);
        struct ggml_tensor * aw = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, F, C);
        struct ggml_tensor * ab = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, F, C);
        struct ggml_tensor * sl = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * rf = ggml_affine_prelu(b.ctx, x, aw, ab, sl);
        double tf = time_variant(&b, rf, (struct ggml_tensor *[]){x, aw, ab, sl},
                                 (float *[]){px, paw, pab, psl}, 4);

        // composed: affine part x*aw[c,f]+ab[c,f] via repeat on F; prelu via
        // max(x,0) + slope*min(x,0) then add.
        bench_begin(&b);
        x  = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F32, F, T, C);
        aw = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, F, C);
        ab = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, F, C);
        sl = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        // broadcast aw/ab from [F, C] to [F, T, C]: view [F,1,C] -> repeat to [F,T,C]
        struct ggml_tensor * aw3 = ggml_reshape_3d(b.ctx, aw, F, 1, C);
        struct ggml_tensor * ab3 = ggml_reshape_3d(b.ctx, ab, F, 1, C);
        struct ggml_tensor * awb = ggml_repeat(b.ctx, aw3, x);
        struct ggml_tensor * abb = ggml_repeat(b.ctx, ab3, x);
        struct ggml_tensor * aff = ggml_add(b.ctx, ggml_mul(b.ctx, x, awb), abb);
        struct ggml_tensor * pos = ggml_relu(b.ctx, x);
        struct ggml_tensor * slb = ggml_repeat(b.ctx, sl, x);
        struct ggml_tensor * neg = ggml_mul(b.ctx, ggml_sub(b.ctx, x, pos), slb);
        struct ggml_tensor * rc  = ggml_add(b.ctx, aff, neg);
        double tc = time_variant(&b, rc, (struct ggml_tensor *[]){x, aw, ab, sl},
                                 (float *[]){px, paw, pab, psl}, 4);

        char name[64]; snprintf(name, sizeof(name), "affine_prelu %dx%dx%d", F, T, C);
        report(name, tf, tc);
        free(px); free(paw); free(pab); free(psl);
    }
}

// ============ 5. channel_shuffle vs permute/reshape/cont ============

static void bench_channel_shuffle(void) {
    printf("\n--- channel_shuffle vs view chain (%s) ---\n", g_backend);
    printf("  %8s %6s %4s %4s | %11s %11s %8s\n", "F", "T", "C", "G", "fused ms", "chain ms", "speedup");

    struct { int F, T, C, G; } cases[] = {
        {16, 256, 64, 8}, {64, 512, 128, 4},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int F = cases[c].F, T = cases[c].T, C = cases[c].C, G = cases[c].G;
        const int cg = C / G;
        float * pa = fill_rand(F*T*C, 51 + (unsigned)c);

        struct bench b; bench_begin(&b);
        struct ggml_tensor * a = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F32, F, T, C);
        struct ggml_tensor * rf = ggml_channel_shuffle(b.ctx, a, G);
        double tf = time_variant(&b, rf, (struct ggml_tensor *[]){a},
                                 (float *[]){pa}, 1);

        // PyTorch channel shuffle over C: [F, T, C] -> reshape [F, T, G, cg]
        //   -> permute to [F, T, cg, G] -> reshape back [F, T, C] -> cont
        bench_begin(&b);
        a = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F32, F, T, C);
        struct ggml_tensor * rs = ggml_reshape_4d(b.ctx, a, F, T, G, C/G);
        // want channel c' = (c%G)*cg + c/G: treat C axis split as (G, cg) then
        // swap to (cg, G) => permutation of dims 2 and 3.
        struct ggml_tensor * pm = ggml_permute(b.ctx, rs, 0, 1, 3, 2);   // [F, T, cg, G]
        struct ggml_tensor * ct = ggml_cont(b.ctx, pm);
        struct ggml_tensor * rc = ggml_reshape_3d(b.ctx, ct, F, T, C);
        double tc = time_variant(&b, rc, (struct ggml_tensor *[]){a},
                                 (float *[]){pa}, 1);

        char name[64]; snprintf(name, sizeof(name), "channel_shuffle %dx%d G%d", F*T, C, G);
        report(name, tf, tc);
        free(pa);
    }
}

// ============ 6. zero_upsample vs upscale(nearest, exact factor) ============

static void bench_zero_upsample(void) {
    printf("\n--- zero_upsample vs composed zero-insertion (%s) ---\n", g_backend);
    printf("  %8s %6s %4s | %11s %11s %8s\n", "F", "R", "s", "fused ms", "chain ms", "speedup");

    struct { int F, R, s; } cases[] = {
        {256, 1, 4}, {1024, 8, 2},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int F = cases[c].F, R = cases[c].R, s = cases[c].s;
        const int Fu = (F - 1)*s + 1;
        float * pa = fill_rand(F*R, 61 + (unsigned)c);

        struct bench b; bench_begin(&b);
        struct ggml_tensor * a = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, F, R);
        struct ggml_tensor * rf = ggml_zero_upsample(b.ctx, a, s);
        double tf = time_variant(&b, rf, (struct ggml_tensor *[]){a},
                                 (float *[]){pa}, 1);

        // composed zero-insertion: upscale (nearest) then mask interior
        // samples to zero via a fixed mask tensor mul.
        double tc = -1;
        bench_begin(&b);
        a = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, F, R);
        struct ggml_tensor * ups = ggml_upscale(b.ctx, a, s, GGML_SCALE_MODE_NEAREST);
        struct ggml_tensor * msk = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, F*s, R);
        struct ggml_tensor * rc  = ggml_mul(b.ctx, ups, msk);
        float * pm = (float *)calloc((size_t)F*s * R, sizeof(float));
        for (int r = 0; r < R; r++)
            for (int f = 0; f < F; f++)
                pm[f*s + r*(F*s)] = 1.0f;

        if (ggml_backend_supports_op(b.backend, rc) && bench_graph(&b, rc)) {
            bench_upload(&b, a, pa);
            bench_upload(&b, msk, pm);
            tc = bench_time_median(&b);
        }
        bench_end(&b);
        free(pm);
        char name[64]; snprintf(name, sizeof(name), "zero_upsample %dx%d s%d", F, R, s);
        report(name, tf, tc);
        free(pa);
    }
}

// ============ 7. supertonic_depthwise_1d vs pad + conv_1d_dw ============

static void bench_supertonic_depthwise(void) {
    printf("\n--- supertonic_depthwise_1d vs pad+conv_1d_dw+add (%s) ---\n", g_backend);
    printf("  %8s %6s %4s %4s | %11s %11s %8s\n", "L", "C", "K", "dil", "fused ms", "chain ms", "speedup");

    struct { int L, C, K, dil; } cases[] = {
        {4096, 64, 7, 1}, {16384, 128, 7, 1},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int L = cases[c].L, C = cases[c].C, K = cases[c].K, dil = cases[c].dil;
        const int p0 = (K / 2) * dil;   // symmetric same-padding for the chain
        float * px = fill_rand(L*C, 71 + (unsigned)c);
        float * pw = fill_rand(K*C, 73 + (unsigned)c);
        float * pb = fill_rand(C, 75 + (unsigned)c);

        struct bench b; bench_begin(&b);
        struct ggml_tensor * x = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, L, C);
        struct ggml_tensor * w = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F32, K, 1, C);
        struct ggml_tensor * bi = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * rf = ggml_supertonic_depthwise_1d(b.ctx, x, w, bi, dil);
        double tf = time_variant(&b, rf, (struct ggml_tensor *[]){x, w, bi},
                                 (float *[]){px, pw, pb}, 3);

        // composed: conv_1d_dw (with p0) + bias add.  The fused op replaces
        // im2col+mul_mat+pad+bias with one pass over [L, C].
        bench_begin(&b);
        x  = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, L, C);
        w  = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F32, K, 1, C);
        bi = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * cv  = ggml_conv_1d_dw(b.ctx, w, x, 1, p0, dil);
        // bias broadcast: cv is [OL, C, 1]
        struct ggml_tensor * bi3 = ggml_reshape_3d(b.ctx, bi, C, 1, 1);
        struct ggml_tensor * bib = ggml_repeat(b.ctx, bi3, cv);
        struct ggml_tensor * rc  = ggml_add(b.ctx, cv, bib);
        double tc = time_variant(&b, rc, (struct ggml_tensor *[]){x, w, bi},
                                 (float *[]){px, pw, pb}, 3);

        char name[64]; snprintf(name, sizeof(name), "depthwise_1d %dx%d K%d", L, C, K);
        report(name, tf, tc);
        free(px); free(pw); free(pb);
    }
}

// ============ 8. supertonic_edge_pad_1d vs concat chain ============

static void bench_supertonic_edge_pad(void) {
    printf("\n--- supertonic_edge_pad_1d vs concat chain (%s) ---\n", g_backend);
    printf("  %8s %6s %4s %4s | %11s %11s %8s\n", "L", "C", "pl", "pr", "fused ms", "chain ms", "speedup");

    struct { int L, C, pl, prr; } cases[] = {
        {4096, 64, 3, 3}, {16384, 128, 7, 7},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int L = cases[c].L, C = cases[c].C, pl = cases[c].pl, prr = cases[c].prr;
        float * px = fill_rand(L*C, 81 + (unsigned)c);

        struct bench b; bench_begin(&b);
        struct ggml_tensor * x  = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, L, C);
        struct ggml_tensor * rf = ggml_supertonic_edge_pad_1d(b.ctx, x, pl, prr);
        double tf = time_variant(&b, rf, (struct ggml_tensor *[]){x},
                                 (float *[]){px}, 1);

        // composed: view first row [1, C] -> repeat pl -> concat with x, then
        // last row similarly on the right.
        bench_begin(&b);
        x = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, L, C);
        struct ggml_tensor * first = ggml_view_2d(b.ctx, x, 1, C, x->nb[1], 0);
        struct ggml_tensor * fr    = ggml_repeat(b.ctx, first, ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, pl, C));
        struct ggml_tensor * lastv = ggml_view_2d(b.ctx, x, 1, C, x->nb[1], (L-1)*sizeof(float));
        struct ggml_tensor * lr    = ggml_repeat(b.ctx, lastv, ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, prr, C));
        struct ggml_tensor * rc    = ggml_concat(b.ctx, ggml_concat(b.ctx, fr, x, 0), lr, 0);
        double tc = time_variant(&b, rc, (struct ggml_tensor *[]){x},
                                 (float *[]){px}, 1);

        char name[64]; snprintf(name, sizeof(name), "edge_pad_1d %dx%d p%d/%d", L, C, pl, prr);
        report(name, tf, tc);
        free(px);
    }
}

// ============ 9. gru (absolute) ============

static void bench_gru(void) {
    printf("\n--- gru absolute (%s) ---\n", g_backend);
    printf("  %6s %4s %5s %7s | %11s\n", "H", "B", "L", "rev", "ms");

    struct { int H, B, L, rev; } cases[] = {
        {64, 1, 256, 0},
        {128, 8, 128, 0},
        {256, 1, 64, 1},
        {512, 4, 32, 1},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int H = cases[c].H, B = cases[c].B, L = cases[c].L, rev = cases[c].rev;
        float * p_whh = fill_rand(H*3*H, 91 + (unsigned)c);
        float * p_gi  = fill_rand(3*H*B*L, 93 + (unsigned)c);
        float * p_bhh = fill_rand(3*H, 95 + (unsigned)c);

        struct bench b; bench_begin(&b);
        struct ggml_tensor * whh = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, H, 3*H);
        struct ggml_tensor * gi  = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F32, 3*H, B, L);
        struct ggml_tensor * bhh = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, 3*H);
        struct ggml_tensor * r   = ggml_gru(b.ctx, whh, gi, bhh, rev != 0);
        double tf = time_variant(&b, r, (struct ggml_tensor *[]){whh, gi, bhh},
                                 (float *[]){p_whh, p_gi, p_bhh}, 3);

        if (tf < 0) printf("  H%-5d B%-4d L%-5d r%d    |  SKIP\n", H, B, L, rev);
        else        printf("  H%-5d B%-4d L%-5d r%d    | %11.4f\n", H, B, L, rev, tf);
        free(p_whh); free(p_gi); free(p_bhh);
    }
}

// ============ 10. supertonic_layer_norm_channel vs permute chain ============

static void bench_supertonic_layer_norm(void) {
    printf("\n--- supertonic_layer_norm_channel vs permute chain (%s) ---\n", g_backend);
    printf("  %8s %6s | %11s %11s %8s\n", "L", "C", "fused ms", "chain ms", "speedup");

    struct { int L, C; } cases[] = {
        {1024, 256}, {4096, 512},
    };

    for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
        const int L = cases[c].L, C = cases[c].C;
        const float eps = 1e-5f;
        float * px = fill_rand(L*C, 101 + (unsigned)c);
        float * pg = fill_rand(C, 103 + (unsigned)c);
        float * pb = fill_rand(C, 107 + (unsigned)c);

        struct bench b; bench_begin(&b);
        struct ggml_tensor * x = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, L, C);
        struct ggml_tensor * g = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * bi = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * rf = ggml_supertonic_layer_norm_channel(b.ctx, x, g, bi, eps);
        double tf = time_variant(&b, rf, (struct ggml_tensor *[]){x, g, bi},
                                 (float *[]){px, pg, pb}, 3);

        // stock chain: permute to [C, L] (C inner-most) -> norm -> affine -> back
        bench_begin(&b);
        x  = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, L, C);
        g  = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        bi = ggml_new_tensor_1d(b.ctx, GGML_TYPE_F32, C);
        struct ggml_tensor * pm  = ggml_permute(b.ctx, x, 1, 0, 2, 3);       // [C, L]
        struct ggml_tensor * cpm = ggml_cont(b.ctx, pm);
        struct ggml_tensor * nm  = ggml_norm(b.ctx, cpm, eps);               // normalizes ne0 = C
        struct ggml_tensor * gb  = ggml_repeat(b.ctx, g, nm);
        struct ggml_tensor * bb  = ggml_repeat(b.ctx, bi, nm);
        struct ggml_tensor * aff = ggml_add(b.ctx, ggml_mul(b.ctx, nm, gb), bb);
        struct ggml_tensor * pb2 = ggml_permute(b.ctx, aff, 1, 0, 2, 3);     // back [L, C]
        struct ggml_tensor * rc  = ggml_cont(b.ctx, pb2);
        double tc = time_variant(&b, rc, (struct ggml_tensor *[]){x, g, bi},
                                 (float *[]){px, pg, pb}, 3);

        char name[64]; snprintf(name, sizeof(name), "LN_channel %dx%d", L, C);
        report(name, tf, tc);
        free(px); free(pg); free(pb);
    }
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc > 1) g_backend = argv[1];
    if (argc > 2) g_threads = atoi(argv[2]);
    printf("== qvac-ops benchmarks (backend: %s, threads: %d, repeats: %d) ==\n",
           g_backend, g_threads, g_repeats);

    bench_snake();
    bench_bias_gelu();
    bench_pw2_residual();
    bench_affine_prelu();
    bench_channel_shuffle();
    bench_zero_upsample();
    bench_supertonic_depthwise();
    bench_supertonic_edge_pad();
    bench_supertonic_layer_norm();
    bench_gru();

    printf("\ndone\n");
    return 0;
}
