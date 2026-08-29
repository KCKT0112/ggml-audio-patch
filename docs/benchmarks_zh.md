# 性能测试

> **[English](benchmarks.md)** | 中文

三后端（CPU / Vulkan / CUDA）微基准，`tests/bench_learned_ops.c` 产出。

## 测试环境

- GPU：NVIDIA GeForce RTX 2070（Turing，CC 7.5，8 GB）
- CPU：x86-64，8 线程，MSVC 2019，AVX2
- 软件栈：ggml v0.19.0（`30bf868`）、CUDA 12.6、Vulkan SDK 1.4.350.0
- 计时：每个用例预热 3 次、计时重复 10 次取均值；ratio = 旧路径耗时 / 新路径耗时，>1 表示新算子更快

## `IM2COL_FAST_1D`（与 `CONV_1D` 逐 shape 对照；shape 列为 `L,IC,OC,K,s,p,d`）

### CPU

| shape | conv_1d | fast_1d | ratio |
|---|---|---|---|
| 100,16,32,3,1,1,1 | 0.0413 ms | 0.0399 ms | **1.03×** |
| 200,32,64,3,1,1,1 | 0.0719 | 0.0685 | **1.05×** |
| 500,64,128,3,2,1,1 | 0.1710 | 0.1617 | **1.06×** |
| 1000,128,256,7,2,3,1 | 3.27 | 2.90 | **1.13×** |
| 2000,256,512,3,1,1,1 | 18.38 | 17.64 | **1.04×** |
| 4000,64,128,5,2,2,1 | 3.17 | 2.76 | **1.15×** |

### Vulkan（ratio）

| | case 1 | case 2 | case 3 |
|---|---|---|---|
| small | 0.97× | 0.98× | 0.86× |
| large | 1.20× | 1.09× | 0.99× |

### CUDA（ratio，两次独立运行）

| | case 1 | case 2 | case 3 |
|---|---|---|---|
| small（run1 / run2） | 0.63 / 0.93 | 1.17 / 1.76 | 1.59 / 1.12 |
| large（run1 / run2） | 0.95 / 1.18 | 0.96 / 1.06 | 0.67 / 0.79 |

### 结论

- **CPU：提速真实且可复现。** 大帧长 1.04–1.15×，小 shape 1.03–1.06×，多次独立运行稳定。属于常量因子级优化：O(1) 窗口消除边界零乘累加，`d0==1` 时中段为 memcpy。与上游类似提案报告的量级一致；此前其它项目报告的更大提速主要来自无对齐 GEMM 的通用 CPU 路径，测试机 x86 走的是对齐 kernel 路径，故收益温和。
- **Vulkan / CUDA：parity。** 该算子在 GPU 后端**别名**到与 `IM2COL` 相同的 im2col+matmul 路径，ratio 围绕 1.0 的波动是测量噪声（亚毫秒 kernel，CUDA 单次 <0.4 ms，两次运行 ±50% 皆常见）。**判定标准是无回归——达标。**

## `conv_transpose_1d` ext（legacy = 6 参 `ggml_conv_transpose_1d`，仅支持 `g=1, p=0, op=0`；shape 列为 `L,Cin,Cout,K,s,p,op,g`）

### CPU

| shape | legacy | ext | ratio |
|---|---|---|---|
| 100,32,64,3,2,0,0,1 | 0.1129 ms | 0.1105 ms | 1.02× |
| 500,16,32,7,1,0,0,1 | 0.1897 | 0.1874 | 1.01× |
| 100,32,64,3,2,0,**2**,1 | – | 0.0967 ms | 新能力（output_padding） |
| 100,32,64,3,2,0,0,**2** | – | 0.1141 ms | 新能力（分组） |
| 200,64,128,5,4,0,1,**4** | – | 0.7460 ms | 新能力（分组） |

### Vulkan

| shape | legacy | ext | ratio |
|---|---|---|---|
| 100,32,64,3,2,0,0,1 | 0.233 ms | 0.108 ms | **2.15×** |
| 500,16,32,7,1,0,0,1 | 0.1013 | 0.0929 | 1.09× |
| op=2 / g=2 / g=4 | – | 0.107 / 0.133 / 0.203 ms | 新能力 |

### CUDA

| shape | legacy | ext | ratio |
|---|---|---|---|
| 100,32,64,3,2,0,0,1（run1） | 0.068 ms | 0.151 ms | 0.45× |
| 同上（run2） | 0.068 | 0.069 | **0.98×** |
| 500,16,32,7,1,0,0,1 | 0.074 | 0.070 | 1.06× |
| op=2 / g=2 / g=4 | – | 0.070 / 0.053 / 0.143 ms | 新能力 |

### 结论

- **功能面**：打 patch 之前，分组（g>1）与 output_padding 在**所有后端**都不可用；ext 补齐后全部正确运行（正确性另有 17 用例测试覆盖）。
- **Vulkan：g=1 可比 case 实测 2.15×。** legacy 路径走 full-im2col（膨胀 ~s0 倍的中间张量）+ matmul；新 kernel 直接按输出坐标反查加权求和，不实例化 im2col。这是本补丁集中最大的单点 GPU 收益。
- **CUDA/CPU：g=1 parity。** CUDA 同一个 case 两次运行给出 0.45× 与 0.98× 两个互相矛盾的比值——亚毫秒 kernel 的测量噪声（见方法论），结论是两者代价同阶。无回归。
- Metal 只接收 `g=1, p=0, d=1` 的超集图通过门控，其余组合回落 CPU。

## `REL_POS_BIAS` / `SCATTER_ELEMENTS`（绝对性能，无旧实现可比）

| 算子 | shape | CPU | Vulkan | CUDA |
|---|---|---|---|---|
| rel_pos_bias | C32 H8 W8 B1 | 2.50 GFLOP/s | 5.1 GFLOP/s | 未移植（正确 SKIP） |
| | C64 H16 W16 B2 | 1.25 | **220.1** | SKIP |
| | C128 H8 W8 B4 | 1.30 | 63.8 | SKIP |
| scatter（覆盖写） | data [1024,1024] ← upd [1024,256] | 0.84 GB/s | 60.4 GB/s | SKIP |
| scatter（累加归约） | data [4096,4096] ← upd [4096,1024] | 0.53 GB/s | **95.9 GB/s**（原子加路径） | SKIP |
| scatter（axis=0） | data [1024,1024] ← upd [256,1024] | 0.77 GB/s | 67.5 GB/s | SKIP |

### 结论

- Vulkan 达到实用水平（60–220 GFLOP/s，60–96 GB/s 带宽级）；scatter 累加因 RTX 2070 支持 `shaderBufferFloat32AtomicAdd` 走了 `atomicAdd` 路径，是三者中最快的。
- CPU 版本是刻意保守的单线程参考实现（`n_tasks=1`），1–2.5 GFLOP/s 只作语义兜底——这两个算子在原生 ggml v0.19.0 中本来就没有 CPU 实现，属于"从无到有"。后续如有需求可并行化。
- CUDA 的 `supports_op` 正确返回 false，上层可干净回落到 CPU。

## `ADD_LEAKY_RELU` / `CONV_DIRECT_1D`（声码器端到端 + 微内核）

> 本节环境：Xeon E5-2675 v3（16C/32T Haswell-EP，AVX2 负载下持续 ~2.0 GHz）、Windows、MSVC 2019 `/O2`、fp32、CPU 后端。消费侧挂具为 pc-nsf-hifigan.cpp @ `f8c16ba` 的 `hifigan_cli`，ggml v0.19.0 + 补丁一，参考音频 T=1722（输出 881 664 采样），`PCNSF_TIMING=1` 墙钟、每路径 3 次取中位。精度对照 torch-CPU fp32 输出（offset 对齐；同机 ONNX Runtime CPU EP：9 次中位 5 155.0 ms）。

微内核，单线程（K=11 形状的 direct-conv 内层循环，每行取最优变体）：

| 变体 | GF/s |
|---|---|
| 纯 FMA 寄存器标定（~2.0 GHz 下 2 FMA/cycle 上限） | 63.4–64.8 |
| 微内核，计数器 `imul` 寻址（最初写法） | 24.6 |
| 微内核，指针递增寻址（**随补丁发布**） | **31.8** |

标定值与最初内核间的 4× 差距来自 `inc → movsxd → imul` 地址链（每 tap 约 5 个串行周期）卡住 `vbroadcastss` 分发；指针递增消除之。距峰值的剩余 2× 是 FMA 关键路径上的广播 load-to-use 延迟——同一循环的寄存器驻留对照变体能跑到标定线，说明瓶颈是延迟而非带宽。

声码器端到端，24 线程（3 次中位）：

| 路径 | 耗时 (ms) | 对 torch-CPU fp32 corr | max\|Δ\| |
|---|---|---|---|
| 原生 `im2col` + `mul_mat`（`PCNSF_DIRECT_CONV=0`） | 80 002 | 0.99999999 | 1.490e-4 |
| direct conv + `ADD_LEAKY_RELU`、无生产侧融合（`PCNSF_FUSE_IO=0`） | 31 772 | 0.99999999 | 1.492e-4 |
| `conv_direct_1d_fused`（输入折入 + 残差 epilogue) | **28 030** | 0.99999999 | 1.492e-4 |

即直接卷积 + 融合相对原生 ggml 卷积路径 **2.85×**。三条路径数值等价（max|Δ| 只在末位有差——FMA 排序噪声）。生产侧融合把计算图从 252 节点减到 152（去掉 50 leaky、5 scale、45 残差 add），输出逐位一致。两点必须如实说明：

- **ggml CPU 路径目前仍比 ONNX Runtime CPU EP 慢约 5.4×**（28.0 s 对 5.155 s）——ORT 的线程化 GEMM 在 Haswell 上调校得好得多。线程扩展也偏弱：融合路径 4 线程约 80 s、24 线程 28 s（6× 线程只换来 2.9×）。缩小该差距（改进分块 / level-0 `[881664, 256]` 级大激活的并行划分）是正在进行的工作；请把 2.85× 读作"赢过原生 ggml"，而非"已与 ORT CPU 同台"。
- *勘误：* 本表旧版曾列 10 038 / 5 973 / 4 764 ms 并宣称与 ORT CPU 持平——那些数字记录于陈旧/不正确的构建配置，不可复现；上表取而代之（同机同模型，逐次重测的 3 次中位）。

同一算子的 Vulkan 路径见下方补丁四一节。

## 方法论与已知陷阱

1. **测试挂具**：多后端 `ggml_backend_sched` 的输出拷贝在该基线版本上有 buffer 混叠嫌疑；本基准采用 **galloc + 单后端 `ggml_backend_graph_compute`**，张量上传/下载显式走 `ggml_backend_tensor_set/get`——与 llama.cpp 单 GPU 主流路径一致。
2. **每个图变体必须有独立的 backend + 分配器生命周期**（begin→graph→upload→计时→end 一套）。在 Vulkan 上复用同一个图分配器顺序处理两个不同 shape 的图会导致访问违例崩溃。
3. **亚毫秒级 GPU kernel 的单次比值噪声可达 ±50%**——CUDA convT 的 0.45×↔0.98× 与 conv small 的 0.63×↔1.76× 都是同一现象。结论一律基于多轮运行的稳定趋势，不以单次为准。
4. legacy convT 断言要求 `g=1, p=0, op=0`，不满足的 shape 只能由 ext 一侧运行（表中 legacy 列标 "–"）。
5. scatter 的 builder 约束 `updates->ne[d] == data->ne[d]（d ≠ axis）`，不合法组合会被断言拒绝。

---

# 补丁二：十个 qvac 融合算子

由 `tests/bench_qvac_ops.c` 产出（`fused` = 新的单分发算子；`composed` = 等价的原版 ggml 子图；`speedup = composed / fused`）。硬件同上（RTX 2070、8 线程 CPU、MSVC 2019 AVX2），ggml v0.19.0 + 两个补丁。**20 次计时取中位数**（3 次热身）——补丁一用的是均值，本组亚毫秒 case 要求中位数。

## 融合 vs 组合（CPU，8 线程）

| case | fused ms | composed ms | speedup |
|---|---|---|---|
| snake 2048×64 | 0.410 | 1.350 | **3.29×** |
| snake 8192×128 | 2.195 | 4.840 | **2.20×** |
| snake 32768×32 | 1.252 | 5.705 | **4.56×** |
| bias_gelu 1024×256 | 0.816 | 1.182 | 1.45× |
| bias_gelu 4096×512 | 6.113 | 10.588 | **1.73×** |
| bias_gelu 1024×1024 | 2.027 | 3.696 | 1.82× |
| pw2_residual 1024×256 | 0.061 | 0.568 | **9.37×** |
| pw2_residual 4096×512 | 0.460 | 4.348 | **9.46×** |
| affine_prelu 64×256×32 | 1.563 | 2.866 | 1.83× |
| affine_prelu 128×512×64 | 12.528 | 20.091 | 1.60× |
| channel_shuffle 4096×64 G8 | 0.055 | 0.047 | 0.85× |
| channel_shuffle 32768×128 G4 | 0.741 | 0.588 | 0.79× |
| zero_upsample 256×1 s4 | 0.009 | 0.026 | 2.78× |
| zero_upsample 1024×8 s2 | 0.010 | 0.141 | **14.87×** |
| depthwise_1d 4096×64 K7 | 0.739 | 21.907 | **29.64×** |
| depthwise_1d 16384×128 K7 | 10.029 | 192.425 | **19.19×** |
| edge_pad_1d 4096×64 p3/3 | 0.151 | 2.039 | **13.51×** |
| edge_pad_1d 16384×128 p7/7 | 1.189 | 15.206 | **12.79×** |
| LN_channel 1024×256 | 0.689 | 1.813 | **2.63×** |
| LN_channel 4096×512 | 15.457 | 12.852 | 0.83× |

（跑了第二轮交叉核对；除 <0.05 ms 的行外，重复性在 ±10% 内。）

### CPU 读数

- **两个 im2col 类融合最猛**：`depthwise_1d`（19–30×）与 `edge_pad_1d`（13×）分别消灭了 F16 im2col scratch 张量和 concat/repeat 拷贝。
- **节点数主导的融合给 2.5–15×**：`pw2_residual`、`zero_upsample`、`snake`——各省掉 2–4 次 kernel 分发。
- **本机上的诚实回退**：`channel_shuffle`（0.79–0.85×）——组合 view 链在连续平面上编译成大 `memcpy`，跑赢了逐平面 gather；`LN_channel` 在 C=512（0.83×）——原版链的 permute 给向量化更友好的最内层 stride。两者在 GPU 分发次数上、以及引擎实际使用的 `[C,T]` 布局上仍是赢家。

## 融合 vs 组合（Vulkan，RTX 2070）

| case | fused ms | composed ms | speedup |
|---|---|---|---|
| snake 2048×64 | 0.109 | 0.180 | 1.65× |
| snake 8192×128 | 0.126 | 0.480 | **3.81×** |
| snake 32768×32 | 0.127 | 0.493 | **3.89×** |
| affine_prelu 64×256×32 | 0.123 | 0.431 | **3.51×** |
| affine_prelu 128×512×64 | 0.276–0.291 | 1.739–1.752 | **6.0–6.3×** |
| channel_shuffle 4096×64 G8 | 0.118 | 0.129 | 1.09× |
| channel_shuffle 32768×128 G4 | 0.285 | 0.229 | 0.80× |
| zero_upsample 256×1 s4 | 0.147 | 0.136 | 0.93× |
| zero_upsample 1024×8 s2 | 0.106 | 0.142 | 1.34× |
| pw2_residual / bias_gelu / depthwise / edge_pad / LN | — | 已测 | fused 未移植 Vulkan（回落 CPU；上游 qvac 为 Metal 内核） |

五个 Vulkan shader 算子符合设计预期：`snake` 与 `affine_prelu`——两个替代"多 kernel 广播链"的——拿到真实 GPU 收益（最高 3.9× 与 6.3×）；拷贝类（`channel_shuffle`、`zero_upsample`）在 GPU 上组合链本就融为单次拷贝，比值在 1.0 附近。

## 融合 vs 组合（Metal，Apple M4）

- 平台：Apple M4（10 核 GPU）、macOS 27.0（26A5421a）、Xcode 27.0（27A5237l）、Apple Clang 21.0.0。
- 软件栈：ggml v0.19.0（`30bf868`）+ 补丁一、补丁二、补丁三。
- 方法：独立运行三个进程。每个进程预热 3 次，对 20 次 `ggml_backend_graph_compute` 计时取中位数；表中再取三轮中位数。基准逐节点检查 `ggml_backend_supports_op`，不包含任何 CPU 回落。

| case | fused ms | composed ms | speedup |
|---|---:|---:|---:|
| snake 2048×64 | 0.241 | 0.282 | 1.17× |
| snake 8192×128 | 0.327 | 0.996 | **3.05×** |
| snake 32768×32 | 0.310 | 0.920 | **2.97×** |
| bias_gelu 1024×256 | 0.256 | 0.288 | 1.13× |
| bias_gelu 4096×512 | 0.458 | 0.880 | 1.92× |
| bias_gelu 1024×1024 | 0.339 | 0.660 | 1.95× |
| pw2_residual 1024×256 | 0.260 | 0.368 | 1.42× |
| pw2_residual 4096×512 | 0.636 | 1.396 | **2.19×** |
| depthwise_1d 4096×64 K7 | 0.241 | 0.655 | **2.72×** |
| depthwise_1d 16384×128 K7 | 0.502 | 2.913 | **5.80×** |
| edge_pad_1d 4096×64 p3/3 | 0.179 | 0.250 | 1.40× |
| edge_pad_1d 16384×128 p7/7 | 0.463 | 0.711 | 1.54× |
| LN_channel 1024×256 | 0.281 | 0.568 | **2.02×** |
| LN_channel 4096×512 | 1.029 | 3.332 | **3.24×** |

最大实测收益是大尺寸 depthwise 的 5.80×；Snake 最高 3.05×，大尺寸通道 layer norm 为 3.24×。edge padding 提速 1.40–1.54×。

供体没有 Metal kernel 的四个融合算子仍不受支持。其纯组合 Metal 基线可以测量：两组形状中，affine-PReLU 为 0.723 / 4.616 ms，channel-shuffle 为 0.375 / 2.148 ms，zero-upsample 为 0.234 / 0.237 ms；由于没有融合 kernel，不宣称加速比。GRU 仍为 SKIP。

## `GRU`（绝对性能；原版无对应实现）

| 后端 | H | B | L | reverse | ms |
|---|---|---|---|---|---|
| CPU | 64 | 1 | 256 | 否 | 4.93 |
| CPU | 128 | 8 | 128 | 否 | 11.11 |
| CPU | 256 | 1 | 64 | 是 | 19.58 |
| CPU | 512 | 4 | 32 | 是 | 47.29 |
| Vulkan (RTX 2070) | 64 | 1 | 256 | 否 | 2.23 |
| Vulkan | 128 | 8 | 128 | 否 | 4.27 |
| Vulkan | H ≥ 256 | | | | SKIP（H ≤ 128 共享内存上限；回落 CPU） |

原版 ggml 没有 RNN 单元——对照列就是这一行的**存在本身**。Vulkan 打包内核（128 lane = 每 workgroup `128/H` 个 batch 元素）在 H ≤ 128 时约为 CPU 的 2×。

## 补丁二方法论备注

6. 挂具规则同补丁一（每变体独立 backend + galloc、不走 sched）。Vulkan 的"每变体独立生命周期"要求再次得到确认：跨图形状复用 subcontext 会崩。
7. Vulkan 上的 `snake` 复用原版 `snake_f32` pipeline（v0.19 基线自带图级 snake 融合、数学一致），fused 列测的是 **op 分发路径**而非新 shader。
8. Vulkan/CPU 上 `channel_shuffle` 的组合链已接近最优单拷贝——speedup < 1 应读作"无收益空间"，不是缺陷。

---

# 补丁四 —— Vulkan `CONV_DIRECT_1D`（声码器端到端）

> 环境：与上文声码器小节同机（Xeon E5-2675 v3，Windows，MSVC 2019），外加 RTX 2070（驱动 32.0.16.2002；warp 32，每 block 48 KiB 共享内存）。软件栈：ggml v0.19.0（`30bf868`）+ 补丁一、二、四（实测时补丁三尚未合入上游；补丁四与 Metal 补丁正交）+ pc-nsf-hifigan.cpp @ `f8c16ba`（`hifigan_cli`，分支 `wip/profile-nodes`）。全程 fp32 并刻意关闭张量核（`GGML_VK_DISABLE_COOPMAT2=1`，`hifigan_cli` 内部默认设置）：声码器契约对齐 torch-CPU 的 fp32 路径，而非 tf32/fp16 张量核算术。计时为 `PCNSF_TIMING=1` 的模型运行墙钟（不含 WAV IO），参考音频 T=1722（约 37.6 s @ 23.5 kHz，输出 881 664 采样）。

端到端（各轮取中位）：

| 路径 | 时间 (ms) | 对 torch-CPU fp32 输出的 corr | max|Δ| |
|---|---|---|---|---|
| ONNX Runtime CPU EP（9 次中位） | 5 155.0 | 0.9999999873 | — |
| ONNX Runtime DML EP（9 次中位） | 325.8 | 0.99999697 | 2.09e-03 |
| ggml Vulkan，补丁四（全部卷积过 supports_op 门控） | **433.2–435.0** | 0.99999985 | 6.13e-04 |
| ggml Vulkan，`PCNSF_DIRECT_CONV=0`（全程回落 im2col） | 571–581 | 0.99999956 | — |

关于该数字的说明：两组 3 次运行的中位为 433.2/435.0（单次 422.6–479.1）。后续在同机做了仅 host 侧的清理（移除 `ggml-vulkan.cpp` 里的调试环境钩子；shader 与图不变，输出**逐位一致**）后复测读数为 458–479 ms（中位 462，n=4）——该偏移是机器状态噪声（数小时编译后的频率/温度漂移），不是代码路径变化：SPIR-V 相同、输出字节相同。诚实引用区间：**≈ 430–480 ms**，DML 级别速度，而 fp32 一致性比 DML 本身紧一个数量级。

分块变体扫描（同一次运行强制各 spec-constant 变体；变体只改分块策略，输出逐位一致）：

| 变体 | e16 | e32 | w64 | w128（发布的默认选择） | e64 |
|---|---|---|---|---|---|
| 中位 ms | 635.6 | 493.9 | 570.1 | 442.0 | 398.2 |

发布的按 OC 选择策略（≤16→e16，≤32→e32，否则 w128）在生产图上**作为整体**最优；一次依据单 shape 微基准提出的 K 感知改选使生产图回退到 488–1 432 ms，已还原。教训记录在案：变体集是协同调参的，单 shape 挂具基准不可外推到全图。

`K ≥ 3` 门控是正确性边界而非偏好：shader 每块暂存 `XS_ROWS = 12` 行输入，`BK = 32` 的通道块跨 `⌊31/K⌋+1` 行输入——K = 2 时为 16 > 11 个可用槽位，循环窗口会覆写仍需的行。生产图有五个 K = 2 亚像素上采样卷积；强制它们上 shader（`PCNSF_DIRECT_MIN_K=1`，绕过消费方门控）会把 corr 确定性地打到 0.36。门控打开时它们回落到同样在 Vulkan 上的 `im2col`+`mul_mat`，成本已包含在上述 433–480 ms 内。

复现（pc-nsf-hifigan.cpp @ `f8c16ba`，`build-vk`，源码树先按序应用补丁一、二、四）：

```bat
set GGML_VK_DISABLE_COOPMAT2=1   rem hifigan_cli 内部已默认
set HF_RAW_OUT=vulkan_out.f32
hifigan_cli.exe hifigan_f32.gguf mel.bin f0.bin out.wav
python cmp_align.py vulkan_out.f32   rem 与 torch-CPU golden 做 offset 对齐 corr
```

21 例正确性挂具（`tools/test_conv_direct.cpp`，含生产 level-0 形状与链式/残差组合）在 Vulkan 路径对 CPU `_fused` 算子 21/21 通过，max|Δ| ≤ 5.5e-5。
