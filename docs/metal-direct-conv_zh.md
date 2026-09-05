# Metal 直接卷积（补丁六）

> **[English](metal-direct-conv.md)** | 中文

本补丁把 pc-nsf-hifigan.cpp（`0abc343`）中的 F32 Metal kernel 整理为可复用的
ggml v0.19.0（`30bf868`）后端补丁。先原样应用集合中的补丁一至五，再从已提交的
ggml 源码差异生成补丁六；它也可直接叠加在补丁一至三之上。

## 范围与接入

`GGML_OP_CONV_DIRECT_1D` 沿用现有七槽参数契约。权重为 `[K, IC, OC]`，输入为
`[T, IC]`，可选 bias 为 `[OC]`，可选 residual 和输出为 `[OL, OC]`，其中
`OL = T + 2*pad - dil*(K-1)`。输入先 scale 再 leaky-ReLU；卷积后依次 bias、
residual、输出 leaky-ReLU。存储和运算保持 F32。浮点累加顺序与 CPU 不同，
不承诺跨后端输出逐位一致。

实现直接把输入 tile 收集到线程组内存，省去 im2col 中间张量。按输出通道数选择
`16×64`、`32×64`、`64×64` tile，K tile 为 8；`OC < 8` 使用每个时间位置一个
scalar 线程。经过 barrier 后复用输入/权重暂存区存放输出累加结果，最大 tile
共享内存从 20 KiB 降至 16 KiB。bias、residual、输入 scale/leaky、输出 leaky
合并在同一次 dispatch 中。

补丁同时把 `IM2COL_FAST_1D` 加入 Metal 原有 im2col 能力判断：补丁一已经提供
dispatch 别名，但遗漏了 `supports_op`。CPU、Vulkan、CUDA kernel 及公共算子
语义均不改变。

Metal 直接卷积要求 `ggml_backend_metal_supports_family(backend, 7)`
（Apple7 simdgroup matrix）、连续 F32 张量、单序列、合法 padding/dilation、
匹配的维度，以及可由 shader 有符号 32 位算术表示的索引。这是 GPU 能力判断，
不是 M 系列名称判断。仅 Apple M4 已真机测试；Intel/AMD Metal GPU 和其他
Apple 设备尚未验证。

**直接执行整张图的调用方必须先检查能力，再选择此图。**
`ggml_backend_graph_compute` 不会自动插入回退。应逐计算节点查询
`ggml_backend_supports_op`，不支持时重建原有 im2col/matmul 图，或者使用带合适
CPU 后端的 scheduler。只检查后端名是否以 `MTL` 开头不够。历史 consumer
pc-nsf-hifigan.cpp `0abc343` 仍有这个限制，手动回退开关为 `PCNSF_DIRECT_CONV=0`。

## 构建与正确性复现

在集合仓库目录执行，同级 `ggml-src` 为干净的 ggml v0.19.0 checkout：

```bash
git -C ../ggml-src checkout 30bf868
git -C ../ggml-src apply "$PWD/patches/learned-ops-ggml0190.patch"
git -C ../ggml-src apply "$PWD/patches/qvac-ops-ggml0190.patch"
git -C ../ggml-src apply "$PWD/patches/metal-ops-ggml0190.patch"
git -C ../ggml-src apply "$PWD/patches/metal-conv-direct-1d-ggml0190.patch"
git -C ../ggml-src apply "$PWD/patches/audio-op-fixes-ggml0190.patch"
git -C ../ggml-src apply "$PWD/patches/cpu-direct-conv-alignment-ggml0190.patch"
GGML_SRC=../ggml-src bash scripts/build-and-test-metal.sh
```

完整集合在三与六之间应用四、五。请使用干净 ggml checkout，不要叠加到 consumer
已经修改过的补丁快照上。嵌入 Metal 源码避免依赖 Xcode 单独的 Metal Toolchain，
kernel 在首次使用时由运行时编译。

脚本构建现有两个测试程序，运行 CPU/Metal CTest，再重复 Metal 测试。参考函数
和误差阈值不变。learned-op 测试现在会在直接执行图前检查全部计算节点，拒绝
本次编译未启用的后端参数，并检查计算返回状态。

2026-09-05 验证环境：Apple M4、macOS 27.0 build 26A5425a、AppleClang 21.0.0、
Release、`GGML_NATIVE=OFF`、`GGML_METAL_EMBED_LIBRARY=ON`：

- CPU/Metal CTest **4/4 通过**；Metal 复跑 **2/2 通过**。 独立
  `GGML_METAL=OFF` 构建的两项 CPU 测试也通过（**2/2**）。
- 直接卷积 **29/29 在 Metal 上实际执行**：原有 17 例，加三种矩阵变体、scalar
  输出、奇数通道数、K=1/2、dilation、时间尾部、padding 与融合组合。
- 拒绝测试覆盖 F16/非连续权重、batch、bias/residual 不匹配、非法
  padding/dilation 和输出长度。
- 现有 qvac CPU/Metal 参考测试通过；不支持的算子显式 SKIP，不记为 GPU 执行。

learned Metal 输出末尾应为：

```text
  case 28: OC=33 IC=13 K=3 OL=65 checked
  done (0 failures so far)
...
[test] Metal direct-conv supports_op rejection cases

ALL PASSED
```

## 历史声码器测量（2026-08-31）

下述数字来自 consumer 基准，不是本次集合打包后新跑的全模型测量。环境为
Apple M4、macOS 27.0、AppleClang 21、Release；OpenVPI
`pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt`、F32 GGUF、1722 帧，输出
881664 个 44100 Hz 采样（19.992 s）。图从 989 节点降至 152 节点，97 个直接
卷积替代 im2col 路径。

| 路径 | 时间 | 重复策略 |
|---|---:|---|
| 原有 Metal im2col/matmul | 6.828–8.911 s | 存档基线区间 |
| 融合 Metal，最快稳定轮 | 0.8360 / 0.8251 / 0.8231 / 0.8228 s | 模型加载一次，连续四次 |
| 融合 Metal，后续轮 | 1.3107 / 1.3501 / 1.3131 / 1.3103 s | 连续四次；独立同源码构建也测得 1.3825 s |
| 融合 Metal，ORT 对比附近 | 中位 1.22085 s | 四次：1.2591 / 1.1924 / 1.2019 / 1.2398 s |
| ORT CPU EP 1.24.1，9 线程 | 中位 3.1560455 s | 预热两次，正式六次，排除 session 创建 |

相近时间窗口中位数比 ORT CPU 快 **2.59×**；最快 Metal 轮为 **23.9–24.3× 实时**。
耗时变化的具体原因未确定，因此保留全部存档轮次。用户自行运行的长样本（16903
帧、输出 196.243 s）耗时 **8.0354 s**，**24.42× 实时**（单次）。特征提取和
模型加载不在 `hifigan_run` 计时范围内。

新 Metal 对 ggml CPU：corr `0.9999999999994206`、max error `2.693e-6`、
RMS `8.919e-8`、p99.9 error `7.562e-7`。对 ORT CPU：corr
`0.9999999999991221`、max error `2.354e-6`、RMS `1.103e-7`。

已否决的调优候选包括 `64×32`、K tile 16、`128×64`、`64×128` 和把
phase-interleave/crop 融入 epilogue：在存档声码器测量中更慢，未包含在补丁内。

在 consumer 复现全模型计时，需要 F32 GGUF，以及同一份 44.1 kHz / hop512 /
128-bin 自然对数 Slaney mel 与逐帧对齐 F0：

```bash
PCNSF_TIMING=1 build-metal/bin/hifigan_cli model.gguf mel.bin f0.bin output.wav
PCNSF_DIRECT_CONV=0 PCNSF_TIMING=1 build-metal/bin/hifigan_cli model.gguf mel.bin f0.bin baseline.wav
```

模型权重和私有音频/特征不随集合发布。
