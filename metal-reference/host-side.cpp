// ============================================================================
// Host-side integration glue for the Metal reference kernels — ggml-audio-patch
// ============================================================================
//
// PROVENANCE (do not remove):
//   Extracted from https://github.com/tetherto/qvac-ext-ggml (branch
//   `speech`, MIT license), files src/ggml-metal/ggml-metal-impl.h,
//   ggml-metal-device.{h,cpp,m}, ggml-metal-ops.{h,cpp}.  Function/struct
//   names and calling conventions are IDENTICAL between the qvac donor tree
//   and the ggml v0.19.0 baseline used by this repo (verified: encoder API
//   names, kargs typedef convention, ops.h / device.h declaration styles,
//   and the erf_approx / SQRT_2_INV dependencies all match).
//
// STATUS: REFERENCE ONLY — not compiled or verified on this repository's
//   development platform.  Integrate piece by piece following
//   docs/metal-porting.md; verify against the acceptance criteria there
//   before flipping any supports_op gate to true.
//
// EDITING RULES (see also /AGENTS.md):
//   - The kargs struct definitions MUST stay field-order-identical to the
//     ones used by the kernels in supertonic_ops.metal (Metal binds
//     `constant` structs positionally).  Field order is part of the ABI
//     between host and shader.
//   - The op_params interpretation below (opts[0..3] slots, layout flags)
//     is frozen by the CPU kernels and tests — do not renumber.
// ============================================================================

// ============================================================================
// PIECE 1 — kargs structs.  Append to src/ggml-metal/ggml-metal-impl.h
// (next to the other ggml_metal_kargs_* typedefs).
// ============================================================================

typedef struct {
    int32_t L;
    int32_t C;
    int32_t K;
    int32_t dilation;
    int32_t has_bias;
    int32_t causal;   // 0 = symmetric edge-clamp (vector_estimator), 1 = causal-left (vocoder)
    int32_t sxt;
    int32_t sxc;
    int32_t syt;
    int32_t syc;
} ggml_metal_kargs_supertonic_depthwise_1d;

typedef struct {
    int32_t L;
    int32_t C;
    float   eps;
    // Per-axis element strides for x and y.  Lets the same kernel handle
    // both [T, C] (sxt=1, sxc=L) and [C, T] (sxt=C, sxc=1) layouts.
    int32_t sxt;  // x stride per time step (in elements)
    int32_t sxc;  // x stride per channel  (in elements)
    int32_t syt;  // y stride per time step (in elements)
    int32_t syc;  // y stride per channel  (in elements)
} ggml_metal_kargs_supertonic_layer_norm_channel;

typedef struct {
    int32_t L;
    int32_t C;
    int32_t sxt;
    int32_t sxc;
    int32_t syt;
    int32_t syc;
    int32_t srt;
    int32_t src;
} ggml_metal_kargs_supertonic_pw2_residual;

typedef struct {
    int32_t L;
    int32_t C;
    int32_t sxt;
    int32_t sxc;
    int32_t syt;
    int32_t syc;
} ggml_metal_kargs_supertonic_bias_gelu;

typedef struct {
    int32_t L_in;
    int32_t L_out;
    int32_t C;
    int32_t pad_left;
    int32_t sxt;
    int32_t sxc;
    int32_t syt;
    int32_t syc;
} ggml_metal_kargs_supertonic_edge_pad_1d;

// snake kargs (name must not collide; the v0.19 baseline has no snake yet)
typedef struct {
    int32_t L;   // ne0 = T
    int32_t C;   // ne1
} ggml_metal_kargs_snake;

// ============================================================================
// PIECE 2 — kernel-name declarations.  Append to src/ggml-metal/ggml-metal.metal
// (the kernels themselves live in supertonic_ops.metal in this directory).
// ============================================================================

// ============================================================================
// PIECE 3 — pipeline lookup functions.  Append to src/ggml-metal/ggml-metal-device.cpp
// and declare in ggml-metal-device.h next to the other get_pipeline_* decls.
// Naming convention matches the baseline: kernel_<name>_<type> is looked up
// (or compiled on first use) through the library cache.
// ============================================================================

#if 0  // reference only — uncomment when integrating

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_SNAKE);

    char base[256];
    snprintf(base, 256, "kernel_snake_%s", ggml_type_name(op->type));

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, base);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, base, nullptr);
    }
    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_supertonic_depthwise_1d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_SUPERTONIC_DEPTHWISE_1D);

    char base[256];
    snprintf(base, 256, "kernel_supertonic_depthwise_1d_%s", ggml_type_name(op->src[0]->type));

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, base);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, base, nullptr);
    }
    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_supertonic_layer_norm_channel(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_SUPERTONIC_LAYER_NORM_CHANNEL);

    char base[256];
    snprintf(base, 256, "kernel_supertonic_layer_norm_channel_%s", ggml_type_name(op->src[0]->type));

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, base);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, base, nullptr);
    }
    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_supertonic_pw2_residual(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_SUPERTONIC_PW2_RESIDUAL);

    char base[256];
    snprintf(base, 256, "kernel_supertonic_pw2_residual_%s", ggml_type_name(op->src[0]->type));

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, base);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, base, nullptr);
    }
    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_supertonic_bias_gelu(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_SUPERTONIC_BIAS_GELU);

    char base[256];
    snprintf(base, 256, "kernel_supertonic_bias_gelu_%s", ggml_type_name(op->src[0]->type));

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, base);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, base, nullptr);
    }
    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_supertonic_edge_pad_1d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_SUPERTONIC_EDGE_PAD_1D);

    char base[256];
    snprintf(base, 256, "kernel_supertonic_edge_pad_1d_%s", ggml_type_name(op->src[0]->type));

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, base);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, base, nullptr);
    }
    return res;
}

#endif // reference only

// ============================================================================
// PIECE 4 — op dispatchers.  Append to src/ggml-metal/ggml-metal-ops.cpp and
// declare in ggml-metal-ops.h next to the other ggml_metal_op_* decls; wire
// into the op switch as `case GGML_OP_X: n_fuse = ggml_metal_op_x(ctx, idx); break;`.
// ============================================================================

#if 0  // reference only — uncomment when integrating

int ggml_metal_op_snake(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t, ne, op, ne);

    const int32_t L = ne0; // T (contiguous inner dim)
    const int32_t C = ne1; // channels

    ggml_metal_kargs_snake args = {
        /*.L =*/ L,
        /*.C =*/ C,
    };

    auto pipeline = ggml_metal_library_get_pipeline_snake(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1); // x
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2); // a
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3); // inv_b
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4); // y

    // Flat 1D grid over T*C elements with a grid-stride loop for high occupancy.
    const int64_t N   = (int64_t) L * (int64_t) C;
    const int     nth = 256;
    int64_t ntg = (N + nth - 1) / nth;
    if (ntg < 1)      ntg = 1;
    if (ntg > 65535)  ntg = 65535;  // grid-stride covers the remainder
    ggml_metal_encoder_dispatch_threadgroups(enc, (int) ntg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_supertonic_depthwise_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne, op, ne);

    // op_params: {K, dilation, layout, causal} (int32 x4).
    //   opts[0]: K in {3, 5, 7}
    //   opts[1]: dilation >= 1
    //   opts[2]: layout flag (0 = [T, C], 1 = [C, T])
    //   opts[3]: causal flag (0 = symmetric edge-clamp, 1 = causal-left pad)
    const int32_t * opts = (const int32_t *) op->op_params;
    const int K        = opts[0];
    const int dilation = opts[1];
    const int32_t layout = opts[2];
    const int32_t causal = opts[3];

    int L, C, sxt, sxc, syt, syc;
    if (layout == 0) {
        L = ne0; C = ne1;
        sxt = 1; sxc = L; syt = 1; syc = L;
    } else {
        C = ne0; L = ne1;
        sxt = C; sxc = 1; syt = C; syc = 1;
    }

    ggml_metal_kargs_supertonic_depthwise_1d args = {
        /*.L        =*/ L,
        /*.C        =*/ C,
        /*.K        =*/ K,
        /*.dilation =*/ dilation,
        /*.has_bias =*/ (op->src[2] != nullptr) ? 1 : 0,
        /*.causal   =*/ causal,
        /*.sxt      =*/ sxt,
        /*.sxc      =*/ sxc,
        /*.syt      =*/ syt,
        /*.syc      =*/ syc,
    };

    auto pipeline = ggml_metal_library_get_pipeline_supertonic_depthwise_1d(lib, op);

    // Cap threads-per-threadgroup at min(L, 1024).  One threadgroup per
    // channel; threads stride over time.
    int nth = L;
    if (nth > 1024) nth = 1024;
    if (nth < 1)    nth = 1;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1); // x
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2); // w
    if (op->src[2] != nullptr) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 3); // bias
    } else {
        // Bind src[0] as a harmless placeholder; the kernel won't read it.
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4); // y

    ggml_metal_encoder_dispatch_threadgroups(enc, C, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_supertonic_layer_norm_channel(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t, ne, op, ne);

    // op_params layout: [eps (f32), layout_flag (i32)].
    float eps;
    memcpy(&eps, op->op_params, sizeof(eps));
    const int32_t layout = ((const int32_t *) op->op_params)[1];

    int L, C;
    int sxt, sxc, syt, syc;
    if (layout == 0) {
        // [T, C]: ne0 = T, ne1 = C.
        L = ne0; C = ne1;
        sxt = 1;       sxc = L;
        syt = 1;       syc = L;
    } else {
        // [C, T]: ne0 = C, ne1 = T.
        C = ne0; L = ne1;
        sxt = C;       sxc = 1;
        syt = C;       syc = 1;
    }

    ggml_metal_kargs_supertonic_layer_norm_channel args = {
        /*.L   =*/ L,
        /*.C   =*/ C,
        /*.eps =*/ eps,
        /*.sxt =*/ sxt,
        /*.sxc =*/ sxc,
        /*.syt =*/ syt,
        /*.syc =*/ syc,
    };

    auto pipeline = ggml_metal_library_get_pipeline_supertonic_layer_norm_channel(lib, op);

    // Threads-per-threadgroup: round up to a multiple of 32 (Apple GPU
    // simdgroup size).  Cap at 256 to limit register pressure.
    int nth = 32;
    while (nth < C && nth < 256) nth *= 2;
    if (nth > C) nth = ((C + 31) / 32) * 32;
    if (nth > 256) nth = 256;
    if (nth < 32) nth = 32;

    // shared scratch: one float per simdgroup, max 8 simdgroups (256/32).
    const size_t shared_bytes = 8 * sizeof(float);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1); // x
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2); // g
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3); // b
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4); // y
    ggml_metal_encoder_set_threadgroup_memory_size(enc, shared_bytes, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, L, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_supertonic_pw2_residual(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t, ne, op, ne);

    // op_params[0]: layout flag.  0 = [T, C] default, 1 = [C, T].
    const int32_t layout = ((const int32_t *) op->op_params)[0];

    int L, C, sxt, sxc, syt, syc, srt, src;
    if (layout == 0) {
        L = ne0; C = ne1;
        sxt = 1; sxc = L; syt = 1; syc = L; srt = 1; src = L;
    } else {
        C = ne0; L = ne1;
        sxt = C; sxc = 1; syt = C; syc = 1; srt = C; src = 1;
    }

    ggml_metal_kargs_supertonic_pw2_residual args = {
        /*.L   =*/ L,
        /*.C   =*/ C,
        /*.sxt =*/ sxt,
        /*.sxc =*/ sxc,
        /*.syt =*/ syt,
        /*.syc =*/ syc,
        /*.srt =*/ srt,
        /*.src =*/ src,
    };

    auto pipeline = ggml_metal_library_get_pipeline_supertonic_pw2_residual(lib, op);

    int nth = L;
    if (nth > 256) nth = 256;
    if (nth < 1)   nth = 1;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1); // x
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2); // bias
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3); // gamma
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4); // residual
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         5); // y

    ggml_metal_encoder_dispatch_threadgroups(enc, C, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_supertonic_bias_gelu(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t, ne, op, ne);

    // op_params[0]: layout flag.  0 = [T, C] default, 1 = [C, T].
    const int32_t layout = ((const int32_t *) op->op_params)[0];

    int L, C, sxt, sxc, syt, syc;
    if (layout == 0) {
        L = ne0; C = ne1;
        sxt = 1; sxc = L; syt = 1; syc = L;
    } else {
        C = ne0; L = ne1;
        sxt = C; sxc = 1; syt = C; syc = 1;
    }

    ggml_metal_kargs_supertonic_bias_gelu args = {
        /*.L   =*/ L,
        /*.C   =*/ C,
        /*.sxt =*/ sxt,
        /*.sxc =*/ sxc,
        /*.syt =*/ syt,
        /*.syc =*/ syc,
    };

    auto pipeline = ggml_metal_library_get_pipeline_supertonic_bias_gelu(lib, op);

    int nth = L;
    if (nth > 256) nth = 256;
    if (nth < 1)   nth = 1;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1); // x
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2); // bias
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3); // y

    ggml_metal_encoder_dispatch_threadgroups(enc, C, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_supertonic_edge_pad_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t, ne, op, ne);

    const int pad_left = ((const int32_t *) op->op_params)[0];
    // opts[2]: layout flag (opts[1] = pad_right).  0 = [T, C], 1 = [C, T].
    const int32_t layout = ((const int32_t *) op->op_params)[2];

    int L_in, L_out, C, sxt, sxc, syt, syc;
    if (layout == 0) {
        L_in  = (int) op->src[0]->ne[0];
        C     = (int) op->src[0]->ne[1];
        L_out = ne0;
        sxt = 1;     sxc = L_in;   syt = 1;     syc = L_out;
    } else {
        C     = (int) op->src[0]->ne[0];
        L_in  = (int) op->src[0]->ne[1];
        L_out = ne1;
        sxt = C;     sxc = 1;      syt = C;     syc = 1;
    }

    ggml_metal_kargs_supertonic_edge_pad_1d args = {
        /*.L_in     =*/ L_in,
        /*.L_out    =*/ L_out,
        /*.C        =*/ C,
        /*.pad_left =*/ pad_left,
        /*.sxt      =*/ sxt,
        /*.sxc      =*/ sxc,
        /*.syt      =*/ syt,
        /*.syc      =*/ syc,
    };

    auto pipeline = ggml_metal_library_get_pipeline_supertonic_edge_pad_1d(lib, op);

    int nth = L_out;
    if (nth > 256) nth = 256;
    if (nth < 1)   nth = 1;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1); // x
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2); // y

    ggml_metal_encoder_dispatch_threadgroups(enc, C, 1, 1, nth, 1, 1);

    return 1;
}

#endif // reference only

// ============================================================================
// PIECE 5 — supports_op gates.  In this patch set, Metal's
// ggml_backend_metal_device_supports_op (src/ggml-metal/ggml-metal-device.m)
// has no cases for these ten ops, so the `default: return false` keeps them
// cleanly on the CPU fallback.  Enable an op ONLY after it passes the full
// verification procedure in docs/metal-porting.md, by adding a case like:
//
//   case GGML_OP_SUPERTONIC_BIAS_GELU:
//       return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
//
// Upstream qvac's gates (for reference — the shapes they accept):
//
//   GGML_OP_SNAKE:                  src0/src1/src2/dst all F32, src0 & dst contiguous
//   GGML_OP_SUPERTONIC_DEPTHWISE_1D: src0/src1/bias F32 (bias may be NULL),
//                                    K in {3, 5, 7} (read via ggml_get_op_params_i32(op, 0))
//   GGML_OP_SUPERTONIC_LAYER_NORM_CHANNEL: src0/src1/src2 F32
//   GGML_OP_SUPERTONIC_PW2_RESIDUAL: src0..src3 F32
//   GGML_OP_SUPERTONIC_BIAS_GELU:    src0/src1 F32
//   GGML_OP_SUPERTONIC_EDGE_PAD_1D:  src0 F32
//
// The other four ops (GRU, ZERO_UPSAMPLE, CHANNEL_SHUFFLE, AFFINE_PRELU)
// have no upstream Metal kernel — leave them gated off.
// ============================================================================
