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

## 方法论与已知陷阱

1. **测试挂具**：多后端 `ggml_backend_sched` 的输出拷贝在该基线版本上有 buffer 混叠嫌疑；本基准采用 **galloc + 单后端 `ggml_backend_graph_compute`**，张量上传/下载显式走 `ggml_backend_tensor_set/get`——与 llama.cpp 单 GPU 主流路径一致。
2. **每个图变体必须有独立的 backend + 分配器生命周期**（begin→graph→upload→计时→end 一套）。在 Vulkan 上复用同一个图分配器顺序处理两个不同 shape 的图会导致访问违例崩溃。
3. **亚毫秒级 GPU kernel 的单次比值噪声可达 ±50%**——CUDA convT 的 0.45×↔0.98× 与 conv small 的 0.63×↔1.76× 都是同一现象。结论一律基于多轮运行的稳定趋势，不以单次为准。
4. legacy convT 断言要求 `g=1, p=0, op=0`，不满足的 shape 只能由 ext 一侧运行（表中 legacy 列标 "–"）。
5. scatter 的 builder 约束 `updates->ne[d] == data->ne[d]（d ≠ axis）`，不合法组合会被断言拒绝。
