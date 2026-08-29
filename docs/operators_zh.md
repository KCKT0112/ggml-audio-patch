# 算子详解与设计笔记

> **[English](operators.md)** | 中文

各算子的语义、API、后端实现方式与出处。

## 1. `GGML_OP_IM2COL_FAST_1D` —— O(1) 窗口的 1D im2col

**出处**：[0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) 的 `ggml_im2col_fast_1d`。

**核心思想**：标准 im2col 对每个输出位置都遍历完整卷积核宽度（O(KW)），其中大量位置落在输入边界外、最终只是零填充。1D 场景下有效窗口可用除法直接算出：

```
base = iow*s0 - p0
ikw0 = ceil(max(0, -base)      / d0)   # 首个有效核位置
ikw1 = ceil(max(0, IW - base)  / d0)   # 窗口结束（上取整封顶）
```

头部 `[0, ikw0)` 与尾部 `[ikw1, KW)` 直接 `memset` 置零；中间有效段在 `d0 == 1` 时是**连续内存，直接 `memcpy`**（F32→F16 目标转换时退化为步进循环）。整体从「每个输出位置扫全核」变为「每个输出位置只碰有效输入」。

**API**：

```c
// 与 ggml_conv_1d 完全同构，仅 im2col 节点改挂 GGML_OP_IM2COL_FAST_1D
struct ggml_tensor * ggml_conv_1d_fast_1d_im2col(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,   // [K, IC, OC]
        struct ggml_tensor  * b,   // [L, IC]
        int s0, int p0, int d0);
```

**后端**：CPU 专用 kernel（`ggml_compute_forward_im2col_fast_1d`，F16/F32 dst × F16/F32 src 全组合）；其余后端一行 case **别名**到既有 `IM2COL` 路径——几何与 op_params 完全同构，别名即可复用全部 GPU 实现。

**实测**：CPU 1.03–1.15×（大帧长收益更大，详见 [benchmarks_zh.md](benchmarks_zh.md)）；GPU 上为别名同路径，parity。

**注意**：ggml 卷积权重布局是 **[K, IC, OC]**——输入通道在 `ne[1]`，与 PyTorch 的 `[OC, IC, K]` 相反。移植/写测试时最常见的错误就是搞反这两个维度。

## 2. `ggml_conv_transpose_1d_ext` —— 全参数转置卷积

**出处**：[mmwillet/TTS.cpp](https://github.com/mmwillet/TTS.cpp) 为其 ggml fork（`support-for-tts` 分支）引入的 `conv_transpose_1d` 修改，服务于 Kokoro 等 TTS 声码器。

**原生 v0.19.0 只有 6 参版本**（无 groups / output_padding / padding）；且上游移植版自带三个 bug，本补丁逐项修复：

1. **CPU 分组路径是错的**——上游注释自认（"the CPU implementation is wrong for groups"）。重写后权重块按 `i1 % cout_pg` 选取、输入通道偏移 `(i1 / cout_pg) * ne02`，与 CUDA / PyTorch 语义对齐。
2. **CUDA 读错 op_params**——原码 `const int p0 = 0; /* opts[3] */`、`const int d0 = 1; /* opts[4] */`：4 参布局 `{s0,p0,d0,g0}` 下正确索引是 1 和 2，被注释掉的 3/4 是旧布局残留。
3. **`GGML_ASSERT(s0 % p0 == 0)` 在 `p0==0` 时除零**——改为 `p0 == 0 || s0 % p0 == 0`。

**新 8 参 API**，旧 6 参保留并委托（`op0=0, g0=1`），二进制向后兼容：

```c
struct ggml_tensor * ggml_conv_transpose_1d_ext(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,    // [K, Cout/g0, Cin/g0]（PyTorch groups 布局）
        struct ggml_tensor  * b,    // [L, Cin]
        int s0, int p0, int d0,     // stride / padding / dilation
        int op0,                    // output_padding
        int g0);                    // groups
```

**布局决策**：groups 采用 PyTorch 布局 `a = [K, Cout/g0, Cin/g0]`——与生态里现成的权重零转换兼容；代价是 `src0->ne[1]` 不再等于输出通道总数，所有把 "src 形状"当 "dst 形状" 用的地方都需要重审（参见 [porting-notes_zh.md](porting-notes_zh.md) 的 Vulkan elements 坑）。

**约束**：`d0 == 1` 且（`p0 == 0 || s0 % p0 == 0`）——各后端 kernel 的公共能力子集。

**后端**：

- **CPU**：f32 与 f16_f32 两条路径都补了分组索引与 p0 越界钳制（`o = i10*s0 + i00 - p0; if (o<0 || o>=ne0) continue;`）。
- **CUDA**：修上述索引 bug，补 `cout_pg/cin_g` 分组索引。
- **Vulkan**：`conv_transpose_1d.comp` 只改索引数学（`Cout_pg_idx = Cout_idx % p.Cout`、`in_c_base = (Cout_idx / p.Cout) * p.Cin`），push-constant 布局不动；同时修正 dispatch 的 elements 为 `dst->ne[1]`。
- **Metal**：kernel 不支持 p0/groups，`supports_op` 在 `g0>1 || p0≠0` 时拒绝，干净回落 CPU。

**实测**：功能补全为主；Vulkan 上即使 g=1 也比 legacy im2col 路径快 **2.15×**（不再实例化 im2col 中间张量）。

## 3. `GGML_OP_REL_POS_BIAS` —— 相对位置偏置

**出处**：[ggmlR](https://CRAN.R-project.org/package=ggmlR)（BoTNet 风格相对位置偏置）。

**语义**：输入 `x = [C, HW, B]`（每幅特征图 HW 个 token 的 C 维特征），权重表 `wcat = [rel_h + rel_w, C]`（`rel_h = 2H-1` 行高差表 + `rel_w = 2W-1` 行宽差表）。输出 `[HW, HW, B]`：

```
out[k, q, b] = Σ_c x[c, q_h·W+q_w, b] · W[r_h(q_h−k_h+H−1), c]
             + Σ_c x[c, q_w·H+q_h, b] · W[rel_h + r_w(q_w−k_w+W−1), c]
```

即「键-查询行列相对位移」各自查一行权重、再与特征做逐通道点积，两轴相加——Swin / BoTNet 一族相对位置注意力的标准偏置生成器。

**后端**：

- **CPU**：直接四重循环 + 通道点积（保守单线程 `n_tasks=1`，语义兜底）。
- **Vulkan**：`rel_pos_bias.comp`，三维 dispatch（x=宽索引, y=高索引, z=b·HW+查询索引），`local_size 8×8×4`，push constants 传 `{H, W, B, C, rel_h, rel_w}`；走 `ggml_vk_op_f32` 通用调度（3 绑定 {x, wcat, dst}）。

**实测**：Vulkan 最高 220 GFLOP/s；CPU 1.2–2.5 GFLOP/s（原生 ggml 此前完全没有该算子的 CPU 实现）。

## 4. `GGML_OP_SCATTER_ELEMENTS` —— 索引散写/累加

**出处**：[ggmlR](https://CRAN.R-project.org/package=ggmlR)（ONNX `ScatterElements` 语义）。

**语义**：`dst = scatter_elements(data, updates, indices, reduction, axis)`。updates 与 indices 同形；对 updates 的每个元素，其在 dst 中的落点 = 同一多维索引、仅沿 `axis` 维替换为 `indices` 的对应值。`reduction=0` 覆盖写，`reduction=1` 累加（重复索引语义由此确定）。本质是 `get_rows` 的逆操作。

**builder 约束**：F32 data/updates、I32 indices、indices 与 updates 形状**逐维一致**（kernel 把两者平铺成同一线性序列索引）、`data` 与 `updates` 仅在 `axis` 维可不同、三张量均连续。**注意：断言必须匹配 kernel 的实际索引方式，而不是只匹配算子规范**——ONNX 允许 indices 广播（维长 1），这里明确拒绝是因为实现并不支持。

**后端**：

- **CPU**：先 memcpy `data→dst`，再平铺遍历 updates，分解 4D 索引、替换 axis 维、写回（`=` 或 `+=`）。
- **Vulkan**：两步——① `buffer_copy` 把 data 拷进 dst + **pipelineBarrier**（`TransferWrite → ShaderRead|ShaderWrite`）；② 按 `vk_op_binary_push_constants` 布局（updates=src0, indices=src1）执行 `scatter_elements.comp`。累加变体使用 `GL_EXT_shader_atomic_float` 的 `atomicAdd`：**设备创建时探测 `VK_EXT_shader_atomic_float` 并检查 `shaderBufferFloat32AtomicAdd`**；不支持的设备上 `supports_op` 返回 false、回落 CPU。

**实测**：Vulkan 60–96 GB/s（累加路径最快，95.9 GB/s）；CPU 0.5–0.8 GB/s 兜底。

## 5. `GGML_OP_ADD_LEAKY_RELU` —— 融合 bias 加法 + leaky ReLU

**出处**：为 [pc-nsf-hifigan.cpp](https://github.com/KakaruHayate/pc-nsf-hifigan.cpp) NSF-HiFiGAN 声码器移植而写：每个 `Conv1d` 后面都跟着 `bias 加 → leaky ReLU`，激活最大到 `[881664, C]`——两个串行的逐元素 pass，各自都要把整张张量重新流一遍。

**语义**：`y = leaky(a + b)`，`leaky(v) = (v > 0 ? v : 0) + slope · (v < 0 ? v : 0)`——与 `ggml_vec_leaky_relu_f32` 表达式完全一致，因此该算子与 `ggml_add` + `ggml_leaky_relu` 组合**逐位一致**（有 parity 测试）。`a = [T, C]`；`b` 为广播 bias `[1, C]` 或逐行 `[T, C]`。

**API**：

```c
struct ggml_tensor * ggml_add_leaky_relu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,      // [T, C]
        struct ggml_tensor  * b,      // [1, C] 或 [T, C]
        float                 slope);
```

**后端**：仅 CPU。`C ≥ nth` 时按通道分条（bias 值驻留寄存器，内层沿连续 t 同时推进两路输入与输出；可自动向量化为两次 select + 乘 + 加），否则按 t 切分。同一改动把原生 `GGML_OP_LEAKY_RELU` 并行化（上游强制 `n_tasks = 1`；补丁按行切分、每元素算术不变）。其他后端 `supports_op` 拒绝 → 干净回落 CPU。

**验证**：8 组用例（广播 `[1,C]` 与逐行 `[T,C]` bias）全部通过精确参考与 add+leaky parity（容差 1e-5/1e-4）。声码器在 im2col 回退路径上使用该算子；下述 direct-conv 路径则把同一模式折进卷积 epilogue。

## 6. `GGML_OP_CONV_DIRECT_1D` —— stride-1 直接 1D 卷积 + 生产侧融合

**出处**：为 pc-nsf-hifigan.cpp NSF-HiFiGAN 声码器 CPU 路径而写。原生 ggml 1D 卷积 = `im2col`（每个节点都物化一遍 `[OL, IC·K]` 的 F16 scratch，即输入 × 核宽的额外字节数）+ 瘦 `OC`、超长形状的 `mul_mat`。直接内核每次调用只打包一次权重，从零填充的输入副本直出输出 tile。

**语义**：stride-1 1D 卷积，可选 bias、输出 leaky ReLU、残差加、输入侧 `leaky(x)·in_scale`：

```
y[t, oc] = act( bias[oc] + Σ_{ic,kw} w[kw, ic, oc] · x'[ic, t + kw·dil] + res[t, oc] )
x'[ic, ·] = 零填充输入行，可选预先折入 in_slope / in_scale
```

所有融合都与未融合图**逐位一致**、不是近似：输入折入在拷贝进填充 scratch 时逐元素应用与 `ggml_vec_leaky_relu_f32` 相同的 leaky 表达式、再乘 `in_scale`（无 fp16 往返）；epilogue 按 `(bias + acc) + res` 结合，与 `ggml_add(conv_out, res)` 一致。

**API**：

```c
struct ggml_tensor * ggml_conv_direct_1d(        // bias + 输出 leaky
        struct ggml_context * ctx,
        struct ggml_tensor  * w,      // [K, IC, OC]
        struct ggml_tensor  * x,      // [T, IC]
        struct ggml_tensor  * bias,   // [OC] 或 NULL
        int pad, int dil,
        float leaky_slope);           // 0 = 无激活

struct ggml_tensor * ggml_conv_direct_1d_fused(  // + 残差、输入侧折入
        struct ggml_context * ctx,
        struct ggml_tensor  * w,      // [K, IC, OC]
        struct ggml_tensor  * x,      // [T, IC]
        struct ggml_tensor  * bias,   // [OC] 或 NULL
        struct ggml_tensor  * res,    // [OL, OC] 或 NULL
        int pad, int dil,
        float leaky_slope,            // 输出激活，0 = 无
        float in_scale,               // 输入折入：v = leaky(x) · in_scale
        float in_slope);              // 输入折入斜率，0 = 无
```

`op_params` = 7 × i32：`{pad, dil, leaky_slope (f32), has_bias, in_scale (f32), in_slope (f32), has_res}`；`src[0..3] = {w, x, bias, res}`；额外工作区 = `(IC·K·OCp + OCp + IC·(T+2·pad))` 个 float。stride 固定为 1——声码器只需要 `s = 1`；一般 stride 走 im2col 路径。

**CPU 内核**（两阶段，均并行）：

- **阶段 1** 把 `Wᵀ` 打包成 `[IC·K, OCp]` 分块布局（`OCp` = OC 向上取整到 16），bias 零填充到 `[OCp]`，把 `x` 拷成 `[IC, T+2·pad]` scratch、两侧边界 `memset` 清零——阶段 2 完全不携带边界判断。生产侧输入折入在此逐元素完成一次。
- **阶段 2** 每线程认领一段均衡的 6 宽 t-block。超级块（oc-超级 × t-超级，按 X 切片 ≤ 64 KB、Wt 切片 ≤ 128 KB 定尺）让两个操作数驻留 L2/L3，X 与 Wt 每次调用大约只从内存流一遍。AVX2 微内核为 6×16 tile（每步 `(ic, kw)` 12 条 FMA + 6 次广播），采用**指针递增寻址**：朴素的循环计数器 `imul` 地址链（`inc → movsxd → imul`，每 tap 约 5 个串行周期）实测会把广播分发卡到 1 FMA/cycle 以下；改成 `xp += dil, wp += OCp` 后单线程从 24.6 提到 31.8 GF/s（+26%，Xeon E5-2675 v3）。标量路径处理尾部 t-block 与非 AVX2 构建。

**后端**：设计上仅 CPU——Vulkan 上 GEMM 更强、保留 im2col 路径；其他后端 `supports_op` 拒绝 → 回落 CPU。

**实测**（Xeon E5-2675 v3，16C/32T Haswell-EP，AVX2 负载 ~2.0 GHz，MSVC 2019 `/O2`，fp32，24 线程，3 次取中位；挂具 pc-nsf-hifigan.cpp @ `f8c16ba`）：NSF-HiFiGAN 全量推理 80 002 ms（原生 im2col + `mul_mat`）→ **28 030 ms**（direct + 生产侧融合），**2.85×**，对 torch-CPU fp32 输出 corr 0.99999999 / max|Δ| 1.49e-4——与 im2col 路径精度同级（仅 FMA 排序噪声）。融合从 252 节点中去掉 100 个（50 leaky、5 scale、45 残差 add）。如实说明差距：ONNX Runtime CPU EP 同模型 5 155 ms——ggml CPU 卷积路径目前落后约 5.4×（ORT 级别的线程化/分块优化是进行中的工作；本段旧版写的 4 764 ms 与 ORT 持平记录于陈旧构建，已更正）。单线程内层循环的指针递增教训依然成立：距线程扩展后峰值的剩余差距来自 FMA 关键路径上的广播 load-to-use 延迟（寄存器驻留对照变体能跑到 2 FMA/cycle 标定线），而非内存带宽。

## 后端支持矩阵（同首页）

| 算子 | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| `IM2COL_FAST_1D` | ✅ 专用 kernel | ✅ 别名 IM2COL | ✅ 别名 | ✅ 别名 |
| `conv_transpose_1d_ext` | ✅ 全参数 | ✅ `p0=0, d0=1`（分组 ✓） | ✅ 全参数 | ⚠️ 仅 `g0=1, p0=0` |
| `REL_POS_BIAS` | ✅ | ✅ | —（回落 CPU） | — |
| `SCATTER_ELEMENTS` | ✅ | ✅（add 需原子扩展） | — | — |
| `ADD_LEAKY_RELU` | ✅ | —（回落 CPU） | — | — |
| `CONV_DIRECT_1D`（含 `_fused`） | ✅ | ✅ 补丁四（fp32，`K ≥ 3`，`(K−1)·dil ≤ 72`，否则回落 CPU） | — | — |

## 上游跟进建议

六个算子都适合作为独立小 patch 向 ggml main 提交讨论：`REL_POS_BIAS` 与 `SCATTER_ELEMENTS` 上游至今空缺；`IM2COL_FAST_1D` 建议附性能数据提交；`conv_transpose_1d_ext` 更适合作为 `GGML_OP_CONV_TRANSPOSE_1D` 的**参数扩展**提案而非新 op。`ADD_LEAKY_RELU` 与 `CONV_DIRECT_1D` 出自声码器需求；direct conv 的指针递增教训（地址链延迟卡住 FMA 广播分发、单线程 +26%）对任何广播-FMA 内积循环都成立，而原生 leaky_relu CPU 内核被强制单线程的问题无论是否收下新算子都值得上游修复。

---

# 补丁二：十个 qvac 融合算子

**来源**：[tetherto/qvac-ext-ggml](https://github.com/tetherto/qvac-ext-ggml) 的 `speech` 分支（MIT）。上游为让三个音频引擎跑成"少分发"计算图而加：**Supertonic** 声码器（ConvNeXt 风格块）、**LavaSR** 降噪器、**ACE-Step Oobleck VAE**。十个都不是包装——每个都把"逐节点调度开销占主导"的多算子子图压成单次遍历（实测见 [benchmarks_zh.md](benchmarks_zh.md)）。

**共同设计**：全部 F32-only（与上游 CPU 实现对齐）、要求连续输入、layout/参数放 `op_params`。五个 supertonic 算子都有 layout-0（`[T,C]`，T 在内）与 layout-1（`_ct`，`[C,T]`，C 在内）两个变体，共享同一内核、靠 stride 翻转区分；`depthwise_1d` 另有 `_causal_ct` 变体（causal 左填充，K ∈ {3,5,7}）。

## 7. `GGML_OP_SUPERTONIC_DEPTHWISE_1D`

```
y[t,c] = bias[c] + Σ_k w[k,c] · x[clamp(t + (k + k_off)·dil, 0, L-1), c]
k_off = causal ? -(K-1) : -K/2
```

把 `pad → im2col → mul_mat → bias`（4 节点 + F16 scratch 张量）压成对 `[L, C]` 的一次遍历。权重布局沿用 ggml 约定 `[K, 1, C]`。CPU 内核按通道分条（bias/w 每通道只载一次）。**CPU 实测比组合 `conv_1d_dw` 链快 19–30×**，主要来自消灭 F16 im2col 物化。

## 8. `GGML_OP_SUPERTONIC_LAYER_NORM_CHANNEL`

```
y[t,c] = (x[t,c] − mean_t) / sqrt(var_t + eps) · g[c] + b[c]
```

原版 `ggml_norm` 只沿 `ne[0]` 归一化，通道轴归一化需要 permute/cont/norm/mul/add/permute/cont 七节点链。单内核完成（mean/var 双精度累加）。CPU 在宽行形状上最高 2.6×；超大 `C` 时 CPU 链的布局反而更快——这正是融合内核在 GPU 端（省 command buffer 条数）价值最大的原因。

## 9. `GGML_OP_SUPERTONIC_PW2_RESIDUAL`

```
y[t,c] = residual[t,c] + (x[t,c] + bias[c]) · gamma[c]
```

三个逐元素算子 → 一个。CPU 3.9–6.8×。

## 10. `GGML_OP_SUPERTONIC_BIAS_GELU`

```
y[t,c] = 0.5·v·(1 + erf(v/√2)),  v = x[t,c] + bias[c]
```

运算次序与 `ggml_gelu_erf` 逐位一致。CPU 1.7–3.2×。

## 11. `GGML_OP_SUPERTONIC_EDGE_PAD_1D`

```
y[t,c] = x[clamp(t − pad_left, 0, L_in − 1), c]
```

复制/边缘钳位 padding（声码器 causal 只填左侧，编码器对称双侧）。替代 view/repeat/concat 链。CPU 12.8–13.5×。

## 12. `GGML_OP_GRU`

融合分批 GRU，一次算完 L 个时间步，PyTorch 语义：门序 r/z/n，reset 作用于 hh（循环）新门，`h0 = 0`。

```
whh [H, 3H]     循环权重（列 g = whh[.., g]）
gi  [3H, B, L]  输入投影（W_ih·x + b_ih 在图里预先算好，不进本算子）
bhh [3H]        循环偏置
dst [H, B, L]
每步：gh = whhᵀh + bhh；r = σ(gi_r + gh_r)；z = σ(gi_z + gh_z)；
      n = tanh(gi_n + r·gh_n)；h = n + z·(h − n)
```

`reverse` 翻转时间方向（BiGRU = 调两次）。此前 ggml 核心**没有任何 RNN 单元**——本算子补上空缺（对 RMVPE 式 BiGRU 后滤波等场景直接可用）。CPU：按 batch 并行、按时间串行、朴素内积。Vulkan：`gru.comp` 每 128-lane workgroup 打包 `128/H` 个 batch 元素（共享内存上限 H ≤ 128），另有 H = 2/4/8 的寄存器驻留 `gru_small.comp` 变体（时间循环零 barrier）。

**实测**：CPU H512×B4×L32 ≈ 47 ms；Vulkan（RTX 2070）H64×B1×L256 ≈ 2.2 ms。

## 13. `GGML_OP_ZERO_UPSAMPLE`

```
out[i0·s, ...] = a[i0, ...]，其余为零；out.ne0 = (a.ne0 − 1)·s + 1
```

零插入上采样——LavaSR 解码器用的转置卷积对偶。替代 `upscale + 掩码乘` 或零填充 convT 技巧。CPU 最高 15×，Vulkan ~1.3×（该规模下受发射次数限制）。

## 14. `GGML_OP_CHANNEL_SHUFFLE`

PyTorch 通道混洗（沿 `ne[2]`）：`in_c = (c' % G)·(C/G) + c'/G`。每输出通道一次平面拷贝，替代 reshape+permute+cont（三节点，其中两个是拷贝）。CPU ~1.5×；Vulkan 上 view 链本就是一次融合拷贝，收益主要是省发射次数。

## 15. `GGML_OP_AFFINE_PRELU`

```
out = x·aw[f,c] + ab[f,c] + max(x,0) + slope[c]·min(x,0)
```

`[F,T,C,Bc]` 频谱形状激活的逐通道仿射 + PReLU（LavaSR 降噪器）。CPU 1.6–2.1×，Vulkan 3.4–6.3×（组合链需要两次 `repeat` 广播 + 四个逐元素内核）。

## 16. `GGML_OP_SNAKE`

```
y = x + sin²(a·x) · inv_b        (a、inv_b 逐通道)
```

ACE-Step Oobleck VAE 的 snake 激活。Vulkan 端本移植**直接复用上游 `snake_f32` pipeline**——基线 v0.19 树里本就有图级 snake 融合（`mul→sin→sqr→mul→add` 模式识别），数学与 binding 布局完全一致，op 路径直接走它（2D `ne0×ne1` 网格，`{ne0, ne1}` push constants）。CPU 2.2–4.6×，Vulkan 最高 3.9×。

## 补丁二后端支持矩阵

| 算子 | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| `SUPERTONIC_DEPTHWISE_1D`（含 `_ct` / `_causal_ct`） | ✅ | —（回落 CPU） | — | — |
| `SUPERTONIC_LAYER_NORM_CHANNEL`（含 `_ct`） | ✅ | — | — | — |
| `SUPERTONIC_PW2_RESIDUAL`（含 `_ct`） | ✅ | — | — | — |
| `SUPERTONIC_BIAS_GELU`（含 `_ct`） | ✅ | — | — | — |
| `SUPERTONIC_EDGE_PAD_1D`（含 `_ct`） | ✅ | — | — | — |
| `GRU` | ✅ | ✅ H ≤ 128（含 H=2/4/8 变体） | — | — |
| `ZERO_UPSAMPLE` | ✅ | ✅ | — | — |
| `CHANNEL_SHUFFLE` | ✅ | ✅ | — | — |
| `AFFINE_PRELU` | ✅ | ✅ | — | — |
| `SNAKE` | ✅ | ✅ | — | — |

上游 qvac 中 supertonic 五件套是 Metal 内核（`kernel_supertonic_*_f32`）、后五件是 Vulkan shader；补丁二范围内本移植保留 Vulkan 五件套、Metal/CUDA 全部关闭（干净回落 CPU）——供体 Metal 内核由补丁三另行接入。

## 补丁二上游跟进建议

`GRU` 是最强的上游候选（RNN 空缺、语义干净、自包含）；`ZERO_UPSAMPLE` + `CHANNEL_SHUFFLE` + `AFFINE_PRELU` + `SNAKE` 可打包成"音频激活/重排"小集合；supertonic 五件套建议等有第二个后端（Metal 或 CUDA）能撑起 API 面积后再作融合算子提案。
