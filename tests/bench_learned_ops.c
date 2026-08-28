// Benchmark: learned-ops vs native ggml implementations.
//
// Usage: bench_learned_ops [cpu|cuda|vulkan]
//
// Measures end-to-end graph_compute time for:
//   1. ggml_conv_1d            vs  ggml_conv_1d_fast_1d_im2col   (1D conv parity + perf)
//   2. ggml_conv_transpose_1d  vs  ggml_conv_transpose_1d_ext   (convT: ext vs 6-arg legacy)
//   3. ggml_rel_pos_bias       (absolute, native has no equivalent)
//   4. ggml_scatter_elements   (absolute, native has no equivalent)
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

#ifdef USE_CUDA
ggml_backend_t             ggml_backend_cuda_init(size_t dev_num);
ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(size_t dev_num);
#endif
#ifdef USE_VULKAN
ggml_backend_t             ggml_backend_vk_init(size_t dev_num);
ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(size_t dev_num);
#endif

static const char * g_backend = "cpu";
static int g_threads = 8;          // CPU thread count
static int g_warmup  = 3;
static int g_repeats = 10;

// format helper: rotating static buffers to allow multiple calls in one printf
static const char * fnum(double v, const char * unit) {
    static char buf[8][64];
    static int  idx = 0;
    int slot = (idx++) & 7;
    snprintf(buf[slot], sizeof(buf[slot]), "%.3f%s", v, unit);
    return buf[slot];
}

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
#ifdef USE_CUDA
    if (strcmp(g_backend, "cuda") == 0) {
        b->backend = ggml_backend_cuda_init(0);
        b->buft = ggml_backend_cuda_buffer_type(0);
        return;
    }
#endif
#ifdef USE_VULKAN
    if (strcmp(g_backend, "vulkan") == 0) {
        b->backend = ggml_backend_vk_init(0);
        b->buft = ggml_backend_vk_buffer_type(0);
        return;
    }
#endif
    b->backend = ggml_backend_cpu_init();
    b->buft = ggml_backend_cpu_buffer_type();
    // set CPU thread count
    ggml_backend_cpu_set_n_threads(b->backend, g_threads);
}

static void bench_graph(struct bench * b, struct ggml_tensor * result) {
    b->gf = ggml_new_graph(b->ctx);
    ggml_build_forward_expand(b->gf, result);
    b->galloc = ggml_gallocr_new(b->buft);
    if (!ggml_gallocr_alloc_graph(b->galloc, b->gf)) {
        printf("  galloc failed\n"); return;
    }
}

static void bench_upload(struct bench * b, struct ggml_tensor * t, const void * host) {
    ggml_backend_tensor_set(t, host, 0, ggml_nbytes(t));
}
static void bench_download(struct bench * b, const struct ggml_tensor * t, void * host) {
    ggml_backend_tensor_get(t, host, 0, ggml_nbytes(t));
}

static double bench_time_compute(struct bench * b) {
    // warmup
    for (int i = 0; i < g_warmup; i++) ggml_backend_graph_compute(b->backend, b->gf);
    // timed
    double t0 = now_s();
    for (int i = 0; i < g_repeats; i++) ggml_backend_graph_compute(b->backend, b->gf);
    double t1 = now_s();
    return (t1 - t0) * 1000.0 / g_repeats;   // ms per iter, mean
}

static void bench_end(struct bench * b) {
    if (b->galloc) ggml_gallocr_free(b->galloc);
    if (b->backend) ggml_backend_free(b->backend);
    ggml_free(b->ctx);
}

static void fill_random(float * p, int n, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        p[i] = (float)((s >> 16) % 2000) * 0.001f - 1.0f;
    }
}

// ============ 1. conv_1d vs conv_1d_fast_1d ============

struct conv_case { int L, IC, OC, K, s0, p0, d0; };

// helper: build+run one graph variant of conv_1d and return mean ms
static double run_conv_case(struct conv_case * cc, int variant, float * pa, float * px) {
    // variant: 0 = ggml_conv_1d, 1 = ggml_conv_1d_fast_1d_im2col
    struct bench b; bench_begin(&b);
    struct ggml_tensor * a = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F32, cc->K, cc->IC, cc->OC);
    struct ggml_tensor * x = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, cc->L, cc->IC);
    struct ggml_tensor * r = variant == 0
        ? ggml_conv_1d(b.ctx, a, x, cc->s0, cc->p0, cc->d0)
        : ggml_conv_1d_fast_1d_im2col(b.ctx, a, x, cc->s0, cc->p0, cc->d0);
    if (!ggml_backend_supports_op(b.backend, r)) {
        bench_end(&b);
        return -1;
    }
    bench_graph(&b, r);
    bench_upload(&b, a, pa); bench_upload(&b, x, px);
    double t = bench_time_compute(&b);
    bench_end(&b);
    return t;
}

static void bench_conv_1d(struct conv_case * cs, int n_cases, const char * label) {
    printf("\n--- %s: conv_1d vs fast_1d (%s) ---\n", label, g_backend);
    printf("  %8s %5s %5s %4s %4s %4s %4s | %10s %10s %7s\n",
           "L","IC","OC","K","s","p","d","conv ms","fast ms","ratio");
    for (int c = 0; c < n_cases; c++) {
        struct conv_case * cc = &cs[c];
        float * pa = malloc(sizeof(float) * (size_t)cc->K * cc->IC * cc->OC);
        float * px = malloc(sizeof(float) * (size_t)cc->L * cc->IC);
        fill_random(pa, (int)(cc->K * cc->IC * cc->OC), 1 + c);
        fill_random(px, (int)(cc->L * cc->IC), 7 + c * 3);
        double t1 = run_conv_case(cc, 0, pa, px);
        double t2 = run_conv_case(cc, 1, pa, px);
        printf("  %8d %5d %5d %4d %4d %4d %4d | %10.3f %10.3f %7.2fx\n",
               cc->L, cc->IC, cc->OC, cc->K, cc->s0, cc->p0, cc->d0, t1, t2,
               (t1 > 0 ? t1/t2 : 0));
        free(pa); free(px);
    }
}

// ============ 2. conv_transpose_1d vs ext ============

struct convT_case { int L, Cin, Cout, K, s0, p0, op0, g0; };

static double run_convT_case(struct convT_case * cc, int variant, float * pa, float * px) {
    // variant: 0 = legacy 6-arg, 1 = ext 8-arg (op0/g0)
    struct bench b; bench_begin(&b);
    struct ggml_tensor * a = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F32, cc->K, cc->Cout/cc->g0, cc->Cin/cc->g0);
    struct ggml_tensor * x = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, cc->L, cc->Cin);
    struct ggml_tensor * r = variant == 0
        ? ggml_conv_transpose_1d(b.ctx, a, x, cc->s0, cc->p0, 1)
        : ggml_conv_transpose_1d_ext(b.ctx, a, x, cc->s0, cc->p0, 1, cc->op0, cc->g0);
    if (!ggml_backend_supports_op(b.backend, r)) {
        bench_end(&b);
        return -1;
    }
    bench_graph(&b, r);
    bench_upload(&b, a, pa); bench_upload(&b, x, px);
    double t = bench_time_compute(&b);
    bench_end(&b);
    return t;
}

static void bench_convT(struct convT_case * cs, int n_cases, const char * label) {
    printf("\n--- %s: conv_transpose_1d (6-arg) vs ext (%s) ---\n", label, g_backend);
    printf("  %6s %5s %5s %4s %4s %4s %4s %4s | %10s %10s %7s\n",
           "L","Cin","Cout","K","s","p","op","g","legacy ms","ext ms","ratio");
    for (int c = 0; c < n_cases; c++) {
        struct convT_case * cc = &cs[c];
        float * pa = malloc(sizeof(float) * (size_t)cc->K * (cc->Cout/cc->g0) * (cc->Cin/cc->g0));
        float * px = malloc(sizeof(float) * (size_t)cc->L * cc->Cin);
        fill_random(pa, (int)(cc->K * (cc->Cout/cc->g0) * (cc->Cin/cc->g0)), 2 + c);
        fill_random(px, (int)(cc->L * cc->Cin), 9 + c * 3);
        // legacy 6-arg only supports g0==1 && p0==0 && op0==0 — skip when unsupported
        double t1 = (cc->g0 == 1 && cc->p0 == 0 && cc->op0 == 0) ? run_convT_case(cc, 0, pa, px) : -1;
        double t2 = run_convT_case(cc, 1, pa, px);
        printf("  %6d %5d %5d %4d %4d %4d %4d %4d | %10s %10s %7s\n",
               cc->L, cc->Cin, cc->Cout, cc->K, cc->s0, cc->p0, cc->op0, cc->g0,
               t1 < 0 ? "SKIP" : fnum(t1,"ms"),
               t2 < 0 ? "SKIP" : fnum(t2,"ms"),
               (t1 > 0 && t2 > 0) ? fnum(t1/t2,"x") : "-");
        free(pa); free(px);
    }
}

// helpers for formatting


// ============ 3. rel_pos_bias (absolute) ============

struct rpb_case { int C, H, W, B; };

static void bench_rel_pos(struct rpb_case * cs, int n_cases) {
    printf("\n--- rel_pos_bias (absolute, %s) ---\n", g_backend);
    printf("  %5s %5s %5s %5s | %10s %12s\n", "C","H","W","B","ms","GFLOP/s est");
    for (int c = 0; c < n_cases; c++) {
        int C = cs[c].C, H = cs[c].H, W = cs[c].W, B = cs[c].B;
        int HW = H*W, Ws = (2*H-1)+(2*W-1);
        struct bench b; bench_begin(&b);
        struct ggml_tensor * x = ggml_new_tensor_3d(b.ctx, GGML_TYPE_F32, C, HW, B);
        struct ggml_tensor * w = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, Ws, C);
        float * px = malloc(ggml_nbytes(x)); fill_random(px, (int)ggml_nelements(x), 3+c);
        float * pw = malloc(ggml_nbytes(w)); fill_random(pw, (int)ggml_nelements(w), 5+c);
        struct ggml_tensor * r = ggml_rel_pos_bias(b.ctx, x, w, H, W);
        bench_graph(&b, r);
        bench_upload(&b, x, px); bench_upload(&b, w, pw);
        double t = (ggml_backend_supports_op(b.backend, r)) ? bench_time_compute(&b) : -1;
        double flops = 2.0 * C * B * HW * HW * 2.0;   // dot products: 2*C per (k,q,b)
        double gflops = (t > 0) ? flops / (t * 1e-3) / 1e9 : 0;
        printf("  %5d %5d %5d %5d | %10s %12.1f\n", C, H, W, B,
               t < 0 ? "SKIP" : fnum(t,"ms"), gflops);
        free(px); free(pw);
        bench_end(&b);
    }
}

// ============ 4. scatter_elements (absolute) ============

struct sc_case { int o0, o1, i0, i1, axis, reduction; }; // data=[o0,o1] upd/idx=[i0,i1]

static void bench_scatter(struct sc_case * cs, int n_cases) {
    printf("\n--- scatter_elements (absolute, %s) ---\n", g_backend);
    printf("  %13s %13s %4s %4s | %10s %12s\n", "data[o0,o1]","upd[i0,i1]","axis","red","ms","GB/s est");
    for (int c = 0; c < n_cases; c++) {
        struct sc_case * cc = &cs[c];
        struct bench b; bench_begin(&b);
        struct ggml_tensor * data = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, cc->o0, cc->o1);
        struct ggml_tensor * upd  = ggml_new_tensor_2d(b.ctx, GGML_TYPE_F32, cc->i0, cc->i1);
        struct ggml_tensor * idx  = ggml_new_tensor_2d(b.ctx, GGML_TYPE_I32, cc->i0, cc->i1);
        float * pd = malloc(ggml_nbytes(data)); fill_random(pd, (int)ggml_nelements(data), 4+c);
        float * pu = malloc(ggml_nbytes(upd));  fill_random(pu, (int)ggml_nelements(upd), 6+c);
        int32_t * pi = malloc(ggml_nbytes(idx));
        // indices range: axis==0 → [0, o0); axis==1 → [0, o1)
        int bound = (cc->axis == 0) ? cc->o0 : cc->o1;
        for (int i = 0; i < cc->i0*cc->i1; i++) pi[i] = (i*7+2) % bound;
        struct ggml_tensor * r = ggml_scatter_elements(b.ctx, data, upd, idx, cc->reduction, cc->axis);
        if (!ggml_backend_supports_op(b.backend, r)) {
            printf("  data=[%5d,%5d] upd=[%5d,%5d] axis=%2d red=%2d |      SKIP\n",
                   cc->o0, cc->o1, cc->i0, cc->i1, cc->axis, cc->reduction);
            free(pd); free(pu); free(pi);
            bench_end(&b);
            continue;
        }
        bench_graph(&b, r);
        bench_upload(&b, data, pd); bench_upload(&b, upd, pu); bench_upload(&b, idx, pi);
        double t = bench_time_compute(&b);
        double bytes = (double)ggml_nbytes(data)*2 + ggml_nbytes(upd) + ggml_nbytes(idx);
        double gbs = (t > 0) ? bytes / (t*1e-3) / 1e9 : 0;
        printf("  data=[%5d,%5d] upd=[%5d,%5d] axis=%2d red=%2d | %10s %12.1f\n",
               cc->o0, cc->o1, cc->i0, cc->i1, cc->axis, cc->reduction,
               fnum(t,"ms"), gbs);
        free(pd); free(pu); free(pi);
        bench_end(&b);
    }
}

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc > 1) g_backend = argv[1];
    if (argc > 2) g_threads = atoi(argv[2]);
    now_s();   // init timer base
    printf("== learned-ops benchmark (backend=%s, threads=%d, warmup=%d, repeats=%d) ==\n",
           g_backend, g_threads, g_warmup, g_repeats);

    // --- 1. conv_1d: shapes spanning TTS/Whisper conv1d stacks ---
    struct conv_case conv_small[] = {
        {100, 16, 32, 3, 1, 1, 1},
        {200, 32, 64, 3, 1, 1, 1},
        {500, 64, 128, 3, 2, 1, 1},
    };
    struct conv_case conv_large[] = {
        {1000, 128, 256, 7, 2, 3, 1},   // Whisper-style
        {2000, 256, 512, 3, 1, 1, 1},
        {4000,  64, 128, 5, 2, 2, 1},
    };
    bench_conv_1d(conv_small, 3, "small");
    bench_conv_1d(conv_large, 3, "large");

    // --- 2. conv_transpose_1d: TTS-style ---
    struct convT_case ct[] = {
        {100, 32, 64, 3, 2, 0, 0, 1},    // plain
        {100, 32, 64, 3, 2, 0, 2, 1},    // op0
        {100, 32, 64, 3, 2, 0, 0, 2},   // groups=2
        {200, 64,128, 5, 4, 0, 1, 4},   // groups=4 large
        {500, 16, 32, 7, 1, 0, 0, 1},   // long kernel
    };
    bench_convT(ct, 5, "TTS-shapes");

    // --- 3. rel_pos_bias ---
    struct rpb_case rpb[] = {
        {32,  8,  8, 1},
        {64, 16, 16, 2},
        {128,8,  8, 4},
    };
    bench_rel_pos(rpb, 3);

    // --- 4. scatter_elements --- (data[o0,o1] vs upd/idx[i0,i1]; non-axis dims must match)
    struct sc_case sc[] = {
        {1024, 1024, 1024,  256, 1, 0},  // axis=1 scatter (row-major inner)
        {4096, 4096, 4096, 1024, 1, 1},  // axis=1 add (atomic on GPU)
        {1024, 1024,  256, 1024, 0, 0},  // axis=0 scatter (whole columns)
    };
    bench_scatter(sc, 3);

    return 0;
}
