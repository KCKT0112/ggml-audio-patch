// ============================================================================
// Metal reference kernels for the ten qvac operators — ggml-audio-patch
// ============================================================================
//
// PROVENANCE (do not remove):
//   Extracted verbatim from https://github.com/tetherto/qvac-ext-ggml
//   (branch `speech`, MIT license), file src/ggml-metal/ggml-metal.metal,
//   in the hope that it helps macOS contributors wire these operators into
//   the Metal backend of ggml v0.19.0.  Upstream reports these kernels as
//   exercised by the QVAC Supertonic / LavaSR / Oobleck engines.
//
// STATUS IN THIS REPO:
//   REFERENCE ONLY — NOT compiled, NOT verified by this repository
//   (development happened on Windows; no macOS runner).  See
//   docs/metal-porting.md for the integration + verification procedure
//   and the acceptance criteria a Metal port must meet before this code
//   moves from "reference" into the patch itself.
//
// EDITING RULES (see also /AGENTS.md):
//   - The kernel bodies below are frozen reference material.  Fix math
//     errors by quoting a failing test case in an issue first; do not
//     silently "simplify" kernels.
//   - Integration glue (kargs structs, dispatch, pipeline lookup,
//     supports_op) you are expected to adapt lives in host-side.cpp —
//     copy from there, not from this file.
//
// DEPENDENCIES AVAILABLE IN THE v0.19 BASELINE (verified):
//   - `SQRT_2_INV` constant and the `erf_approx<T>` template already exist
//     in the baseline ggml-metal.metal (used by kernel_gelu_erf_f32), so
//     kernel_supertonic_bias_gelu_f32 compiles as-is.
//
// Missing from this file on purpose: GRU / ZERO_UPSAMPLE / CHANNEL_SHUFFLE /
// AFFINE_PRELU — upstream qvac does not ship Metal kernels for those four;
// they are CPU (+Vulkan) only in both projects.  The Vulkan shaders in the
// patch (gru.comp etc.) are better starting points for a Metal port of those.
// ============================================================================

// ----------------------------------------------------------------------------
// 1. snake activation (GGML_OP_SNAKE)
//    y = x + sin^2(a*x) * inv_b, per-channel a / inv_b (ACE-Step Oobleck VAE).
//    x / y are [T, C] contiguous; one threadgroup grid over all elements,
//    grid-stride loop.  Mirrors ggml_compute_forward_snake.
// ----------------------------------------------------------------------------

kernel void kernel_snake_f32(
        constant ggml_metal_kargs_snake & args,
        device const float * x,      // [T, C]
        device const float * a,      // [C]
        device const float * inv_b,  // [C]
        device       float * y,      // [T, C]
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg[[threadgroups_per_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {

    const uint L = (uint) args.L;
    const uint C = (uint) args.C;
    const uint N = L * C;

    // Flat 1D grid over all T*C elements (grid-stride).  x/y are contiguous
    // [T, C] so element index == t + c*L; the per-channel scalars come from c.
    const uint stride = ntg.x * tgpg.x;
    for (uint gid = tgpig.x * ntg.x + tpitg.x; gid < N; gid += stride) {
        const uint  c  = gid / L;
        const float xi = x[gid];
        const float si = sin(a[c] * xi);
        y[gid] = xi + si * si * inv_b[c];
    }
}

// ----------------------------------------------------------------------------
// 2. supertonic fused depthwise-1D conv (GGML_OP_SUPERTONIC_DEPTHWISE_1D)
//    edge-clamp (or causal-left) padding + depthwise conv + bias in one pass.
//    One threadgroup per channel; threads iterate over the time dimension.
//
//    Layout (all f32 contiguous):
//      x:    [L, C, 1, 1] memory offset (t, c) -> c*L + t   (layout 0)
//      w:    [K, 1, C, 1] memory offset (k, 0, c, 0) -> c*K + k
//      bias: [C]          memory offset c -> c     (omitted when has_bias = 0)
//      y:    [L, C, 1, 1] same layout as x
// ----------------------------------------------------------------------------

kernel void kernel_supertonic_depthwise_1d_f32(
    constant   ggml_metal_kargs_supertonic_depthwise_1d & args,
    device  const float * x,
    device  const float * w,
    device  const float * bias,
    device        float * y,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int c = (int) tgpig.x;
    if (c >= args.C) return;

    const int L        = args.L;
    const int K        = args.K;
    const int dilation = args.dilation;
    const int causal   = args.causal;
    const int sxt = args.sxt, sxc = args.sxc;
    const int syt = args.syt, syc = args.syc;

    // Layout-agnostic per-channel base pointers: x[t, c] = x[t*sxt + c*sxc].
    // w is stored as [K, 1, C] f32 contiguous, so w_c = w + c*K stays.
    device const float * x_c = x + (size_t) c * sxc;
    device const float * w_c = w + (size_t) c * K;
    device       float * y_c = y + (size_t) c * syc;

    const float bias_v = (args.has_bias != 0) ? bias[c] : 0.0f;

    // The k-offset selects the kernel-centre vs causal-left convolution
    // semantic: symmetric edge-clamp (causal=0) uses offset = -K/2 so the
    // tap at k = K/2 lands at t; causal (causal=1) uses offset = -(K-1) so
    // the last tap at k = K-1 lands at t and earlier taps look strictly
    // left.
    const int k_off = (causal != 0) ? -(K - 1) : -(K / 2);

    for (int t = (int) tpitg.x; t < L; t += (int) ntg.x) {
        float sum = bias_v;
        // Compile-time peeled inner loop for K in {3, 5, 7}.  K=3/5 is the
        // vector_estimator's symmetric ConvNeXt; K=7 is the vocoder's causal
        // ConvNeXt.  Right-clamp `s >= L` is required for the symmetric path
        // only — in causal mode all taps satisfy s <= t < L by construction.
        if (K == 7) {
            int s0 = t + (0 + k_off)*dilation; if (s0 < 0) s0 = 0; else if (s0 >= L) s0 = L - 1;
            int s1 = t + (1 + k_off)*dilation; if (s1 < 0) s1 = 0; else if (s1 >= L) s1 = L - 1;
            int s2 = t + (2 + k_off)*dilation; if (s2 < 0) s2 = 0; else if (s2 >= L) s2 = L - 1;
            int s3 = t + (3 + k_off)*dilation; if (s3 < 0) s3 = 0; else if (s3 >= L) s3 = L - 1;
            int s4 = t + (4 + k_off)*dilation; if (s4 < 0) s4 = 0; else if (s4 >= L) s4 = L - 1;
            int s5 = t + (5 + k_off)*dilation; if (s5 < 0) s5 = 0; else if (s5 >= L) s5 = L - 1;
            int s6 = t + (6 + k_off)*dilation; if (s6 < 0) s6 = 0; else if (s6 >= L) s6 = L - 1;
            sum += x_c[(size_t) s0 * sxt] * w_c[0]
                 + x_c[(size_t) s1 * sxt] * w_c[1]
                 + x_c[(size_t) s2 * sxt] * w_c[2]
                 + x_c[(size_t) s3 * sxt] * w_c[3]
                 + x_c[(size_t) s4 * sxt] * w_c[4]
                 + x_c[(size_t) s5 * sxt] * w_c[5]
                 + x_c[(size_t) s6 * sxt] * w_c[6];
        } else if (K == 5) {
            int s0 = t + (0 + k_off)*dilation; if (s0 < 0) s0 = 0; else if (s0 >= L) s0 = L - 1;
            int s1 = t + (1 + k_off)*dilation; if (s1 < 0) s1 = 0; else if (s1 >= L) s1 = L - 1;
            int s2 = t + (2 + k_off)*dilation; if (s2 < 0) s2 = 0; else if (s2 >= L) s2 = L - 1;
            int s3 = t + (3 + k_off)*dilation; if (s3 < 0) s3 = 0; else if (s3 >= L) s3 = L - 1;
            int s4 = t + (4 + k_off)*dilation; if (s4 < 0) s4 = 0; else if (s4 >= L) s4 = L - 1;
            sum += x_c[(size_t) s0 * sxt] * w_c[0]
                 + x_c[(size_t) s1 * sxt] * w_c[1]
                 + x_c[(size_t) s2 * sxt] * w_c[2]
                 + x_c[(size_t) s3 * sxt] * w_c[3]
                 + x_c[(size_t) s4 * sxt] * w_c[4];
        } else { // K == 3
            int s0 = t + (0 + k_off)*dilation; if (s0 < 0) s0 = 0; else if (s0 >= L) s0 = L - 1;
            int s1 = t + (1 + k_off)*dilation; if (s1 < 0) s1 = 0; else if (s1 >= L) s1 = L - 1;
            int s2 = t + (2 + k_off)*dilation; if (s2 < 0) s2 = 0; else if (s2 >= L) s2 = L - 1;
            sum += x_c[(size_t) s0 * sxt] * w_c[0]
                 + x_c[(size_t) s1 * sxt] * w_c[1]
                 + x_c[(size_t) s2 * sxt] * w_c[2];
        }
        y_c[(size_t) t * syt] = sum;
    }
}

// ----------------------------------------------------------------------------
// 3. supertonic fused channel-axis layer norm
//    (GGML_OP_SUPERTONIC_LAYER_NORM_CHANNEL)
//    Replaces permute + cont + ggml_norm + mul + add + permute + cont.
//    One threadgroup per timestep; threads stripe over channels.
//    Reduction: simdgroup_sum + threadgroup memory across simdgroups.
//    Assumes threads_per_threadgroup is a multiple of 32 (Apple simdgroup
//    size).  Caller picks nth = min(L, 32 * ceil(C / 32)) up to 256.
// ----------------------------------------------------------------------------

kernel void kernel_supertonic_layer_norm_channel_f32(
    constant   ggml_metal_kargs_supertonic_layer_norm_channel & args,
    device  const float * x,
    device  const float * g,
    device  const float * b,
    device        float * y,
    threadgroup    float * shared [[threadgroup(0)]],
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]],
    uint  tiisg [[thread_index_in_simdgroup]]) {

    const int t = (int) tgpig.x;
    if (t >= args.L) return;

    const int C = args.C;
    // Element strides — let the same kernel handle both
    // [T, C] (sxt=1, sxc=L) and [C, T] (sxt=C, sxc=1) layouts.
    const int sxt = args.sxt, sxc = args.sxc;
    const int syt = args.syt, syc = args.syc;

    device const float * x_t = x + (size_t) t * sxt;
    device       float * y_t = y + (size_t) t * syt;

    // ---- Mean ----
    float my_sum = 0.0f;
    for (int c = (int) tpitg.x; c < C; c += (int) ntg.x) {
        my_sum += x_t[(size_t) c * sxc];
    }
    // Simdgroup reduce within the simdgroup.
    my_sum = simd_sum(my_sum);
    if (tiisg == 0) {
        shared[sgitg] = my_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // First simdgroup reduces partial sums.
    if (sgitg == 0) {
        const uint n_sg = (ntg.x + 31) / 32;
        float total = (tiisg < n_sg) ? shared[tiisg] : 0.0f;
        total = simd_sum(total);
        if (tiisg == 0) shared[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = shared[0] / (float) C;

    // ---- Variance ----
    float my_sq = 0.0f;
    for (int c = (int) tpitg.x; c < C; c += (int) ntg.x) {
        const float d = x_t[(size_t) c * sxc] - mean;
        my_sq += d * d;
    }
    my_sq = simd_sum(my_sq);
    if (tiisg == 0) {
        shared[sgitg] = my_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        const uint n_sg = (ntg.x + 31) / 32;
        float total = (tiisg < n_sg) ? shared[tiisg] : 0.0f;
        total = simd_sum(total);
        if (tiisg == 0) shared[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_std = rsqrt(shared[0] / (float) C + args.eps);

    // ---- Apply affine ----
    for (int c = (int) tpitg.x; c < C; c += (int) ntg.x) {
        const float xv = x_t[(size_t) c * sxc];
        y_t[(size_t) c * syc] = (xv - mean) * inv_std * g[c] + b[c];
    }
}

// ----------------------------------------------------------------------------
// 4. supertonic fused (x + bias) * gamma + residual
//    (GGML_OP_SUPERTONIC_PW2_RESIDUAL)
//    One threadgroup per channel; threads stride over the time dim; bias and
//    gamma live in registers for the whole row.
// ----------------------------------------------------------------------------

kernel void kernel_supertonic_pw2_residual_f32(
    constant   ggml_metal_kargs_supertonic_pw2_residual & args,
    device  const float * x,
    device  const float * bias,
    device  const float * gamma,
    device  const float * residual,
    device        float * y,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int c = (int) tgpig.x;
    if (c >= args.C) return;

    const int L = args.L;
    const int sxt = args.sxt, sxc = args.sxc;
    const int syt = args.syt, syc = args.syc;
    const int srt = args.srt, src = args.src;
    const float bv = bias[c];
    const float gv = gamma[c];

    device const float * x_c = x        + (size_t) c * sxc;
    device const float * r_c = residual + (size_t) c * src;
    device       float * y_c = y        + (size_t) c * syc;

    for (int t = (int) tpitg.x; t < L; t += (int) ntg.x) {
        y_c[(size_t) t * syt] = r_c[(size_t) t * srt] + (x_c[(size_t) t * sxt] + bv) * gv;
    }
}

// ----------------------------------------------------------------------------
// 5. supertonic fused bias + GELU (GGML_OP_SUPERTONIC_BIAS_GELU)
//    y[t, c] = gelu_erf(x[t, c] + bias[c])
//            = 0.5 * v * (1 + erf(v / sqrt(2)))
//    Uses the baseline's erf_approx template (same one kernel_gelu_erf_f32
//    uses) so the fused output matches the unfused add + gelu_erf path.
//    One threadgroup per channel; threads stride over the time axis.
// ----------------------------------------------------------------------------

kernel void kernel_supertonic_bias_gelu_f32(
    constant   ggml_metal_kargs_supertonic_bias_gelu & args,
    device  const float * x,
    device  const float * bias,
    device        float * y,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int c = (int) tgpig.x;
    if (c >= args.C) return;

    const int L = args.L;
    const int sxt = args.sxt, sxc = args.sxc;
    const int syt = args.syt, syc = args.syc;
    const float bv = bias[c];

    device const float * x_c = x + (size_t) c * sxc;
    device       float * y_c = y + (size_t) c * syc;

    for (int t = (int) tpitg.x; t < L; t += (int) ntg.x) {
        const float v = x_c[(size_t) t * sxt] + bv;
        y_c[(size_t) t * syt] = 0.5f * v * (1.0f + erf_approx<float>(v * SQRT_2_INV));
    }
}

// ----------------------------------------------------------------------------
// 6. supertonic edge-replicate 1D padding (GGML_OP_SUPERTONIC_EDGE_PAD_1D)
//    y[t, c] = x[clamp(t - pad_left, 0, L_in - 1), c]; output
//    [L_in + pad_left + pad_right, C, 1, 1].  One threadgroup per channel;
//    threads stride over the (longer) output time axis.
// ----------------------------------------------------------------------------

kernel void kernel_supertonic_edge_pad_1d_f32(
    constant   ggml_metal_kargs_supertonic_edge_pad_1d & args,
    device  const float * x,
    device        float * y,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int c = (int) tgpig.x;
    if (c >= args.C) return;

    const int L_in  = args.L_in;
    const int L_out = args.L_out;
    const int pad   = args.pad_left;
    const int sxt = args.sxt, sxc = args.sxc;
    const int syt = args.syt, syc = args.syc;

    device const float * x_c = x + (size_t) c * sxc;
    device       float * y_c = y + (size_t) c * syc;

    for (int t = (int) tpitg.x; t < L_out; t += (int) ntg.x) {
        int src_t = t - pad;
        if (src_t < 0)      src_t = 0;
        if (src_t >= L_in)  src_t = L_in - 1;
        y_c[(size_t) t * syt] = x_c[(size_t) src_t * sxt];
    }
}
