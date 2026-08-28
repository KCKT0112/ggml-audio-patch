# 算子详解与设计笔记

> **[English](operators.md)** | 中文

四个算子的语义、API、后端实现方式与出处。

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

## 后端支持矩阵（同首页）

| 算子 | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| `IM2COL_FAST_1D` | ✅ 专用 kernel | ✅ 别名 IM2COL | ✅ 别名 | ✅ 别名 |
| `conv_transpose_1d_ext` | ✅ 全参数 | ✅ `p0=0, d0=1`（分组 ✓） | ✅ 全参数 | ⚠️ 仅 `g0=1, p0=0` |
| `REL_POS_BIAS` | ✅ | ✅ | —（回落 CPU） | — |
| `SCATTER_ELEMENTS` | ✅ | ✅（add 需原子扩展） | — | — |

## 上游跟进建议

四个算子都适合作为独立小 patch 向 ggml main 提交讨论：`REL_POS_BIAS` 与 `SCATTER_ELEMENTS` 上游至今空缺；`IM2COL_FAST_1D` 建议附性能数据提交；`conv_transpose_1d_ext` 更适合作为 `GGML_OP_CONV_TRANSPOSE_1D` 的**参数扩展**提案而非新 op。
