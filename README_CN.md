# ggml-audio-patch

> 中文 | **[English](README.md)**

一套精选补丁，把来自 ggml 生态不同项目的**十四个音频域算子**统一移植进 [ggml](https://github.com/ggml-org/ggml) **v0.19.0**：API 对齐上游规范、修复了原生实现的 bug、并按后端能力补齐 CPU / Vulkan / CUDA 支持。以两个顺序应用的统一 diff + 正确性测试 + 跨后端基准程序交付。

## 补丁一：四个 learned 算子

| 算子 | 学习来源 | 价值 |
|---|---|---|
| `GGML_OP_IM2COL_FAST_1D` | [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) | 1D im2col 的每行有效窗口 O(1) 直接算出，不再扫完整核宽；`d0 == 1` 时窗口中段退化为纯 `memcpy`。 |
| `ggml_conv_transpose_1d_ext` | [mmwillet/TTS.cpp](https://github.com/mmwillet/TTS.cpp)（ggml `support-for-tts` 分支） | 与 PyTorch `ConvTranspose1d` 全参数对齐：**groups / output_padding / padding**；并修复其损坏的 CPU 分组路径、CUDA `op_params` 读错位与除零断言。 |
| `GGML_OP_REL_POS_BIAS` | [ggmlR](https://CRAN.R-project.org/package=ggmlR)（ggml 的 R 绑定） | BoTNet 风格双轴相对位置注意力偏置：按轴查位移表 + 逐通道点积，含 CPU 与 Vulkan 实现。 |
| `GGML_OP_SCATTER_ELEMENTS` | [ggmlR](https://CRAN.R-project.org/package=ggmlR) | ONNX `ScatterElements` 语义——`get_rows` 的逆操作。Vulkan 端用 `VK_EXT_shader_atomic_float` 原子加实现累加归约。 |

## 补丁二：十个 qvac 融合算子

移植自 [tetherto/qvac-ext-ggml](https://github.com/tetherto/qvac-ext-ggml)（`speech` 分支，MIT 许可），**叠加在补丁一之上**。十个全部是真实融合内核——把原版 ggml 需要 3–5 个独立算子表达、逐节点调度开销占主导的子图压成单次遍历（且原版 ggml **完全没有 RNN 单元**，`gru` 补上了这个空缺）：

| 算子 | 来源引擎 | 融合内容 |
|---|---|---|
| `GGML_OP_SUPERTONIC_DEPTHWISE_1D` | Supertonic 声码器（ConvNeXt 块） | pad（edge-clamp 或 causal）+ 逐通道 1D 卷积 + bias 一次完成；`[T,C]` / `[C,T]` 双布局，K ∈ {3,5,7}，支持 dilation。 |
| `GGML_OP_SUPERTONIC_LAYER_NORM_CHANNEL` | Supertonic 声码器 | 通道轴 layer norm（原版 `ggml_norm` 只归一化 `ne[0]`）+ 仿射，替代 permute/cont/norm/mul/add/permute/cont 七节点链。 |
| `GGML_OP_SUPERTONIC_PW2_RESIDUAL` | Supertonic 声码器 | `(x + bias) * gamma + residual` 单次完成。 |
| `GGML_OP_SUPERTONIC_BIAS_GELU` | Supertonic 声码器 | bias 加法 + erf-GELU 单次完成。 |
| `GGML_OP_SUPERTONIC_EDGE_PAD_1D` | Supertonic 声码器 | 边缘复制 padding（仅左侧或对称双侧），替代 view/repeat/concat 链。 |
| `GGML_OP_GRU` | LavaSR 降噪器 | 融合分批 GRU 全时间步扫描（PyTorch 语义，门序 r/z/n），按 batch 并行——ggml 核心首个 RNN 单元。 |
| `GGML_OP_ZERO_UPSAMPLE` | LavaSR 降噪器 | 整数倍零插入上采样（转置卷积的对偶），单次遍历。 |
| `GGML_OP_CHANNEL_SHUFFLE` | LavaSR 降噪器 | PyTorch 通道混洗（沿 `ne[2]`），每输出通道一次平面拷贝。 |
| `GGML_OP_AFFINE_PRELU` | LavaSR 降噪器 | 逐通道仿射 + PReLU 单次完成。 |
| `GGML_OP_SNAKE` | ACE-Step Oobleck VAE | snake 激活 `y = x + sin²(a·x)·inv_b`，逐通道参数。 |

五个 supertonic 算子此处仅 CPU 实现（上游 qvac 为 Metal 内核，Metal 移植留待后续）；`GRU` / `ZERO_UPSAMPLE` / `CHANNEL_SHUFFLE` / `AFFINE_PRELU` / `SNAKE` 五个另有 Vulkan compute shader 实现，`GRU` 额外带 H = 2/4/8 的寄存器驻留变体，共享内存上限 H ≤ 128。

基线：ggml [`30bf868`](https://github.com/ggml-org/ggml)（v0.19.0）。diff 只在枚举/builder/kernel 的插入点上做增量，应用到邻近 commit 通常只需少量冲突处理。

## 目录结构

```
ggml-audio-patch/
├── patches/
│   ├── learned-ops-ggml0190.patch   # 基于 ggml v0.19.0 的统一 diff（补丁一）
│   └── qvac-ops-ggml0190.patch      # 叠加在补丁一之上的统一 diff（补丁二）
├── tests/
│   ├── test_learned_ops.c           # 补丁一正确性冒烟测试（手写参考值对照）
│   ├── test_qvac_ops.c              # 补丁二正确性测试（CPU 全部 10 个；Vulkan 5 个 shader 算子）
│   ├── bench_learned_ops.c          # 补丁一 CPU / Vulkan / CUDA 微基准
│   └── bench_qvac_ops.c             # 补丁二 融合 vs 组合图 基准（CPU + Vulkan）
├── scripts/
│   ├── build-and-test.sh            # Linux/macOS 一键构建+测试
│   └── build-and-test.ps1           # Windows (pwsh) 一键构建+测试
└── docs/
    ├── building.md / building_zh.md            # 构建须知
    ├── benchmarks.md / benchmarks_zh.md        # 性能测试与方法论
    ├── operators.md / operators_zh.md          # 各算子设计笔记与 API
    └── porting-notes.md / porting-notes_zh.md  # 移植坑与修复记录
```

## 快速开始

```bash
git clone https://github.com/ggml-org/ggml.git ggml-src
cd ggml-src && git checkout 30bf868        # v0.19.0
git apply ../ggml-audio-patch/patches/learned-ops-ggml0190.patch   # 补丁一
git apply ../ggml-audio-patch/patches/qvac-ops-ggml0190.patch      # 补丁二（顺序应用）
```

补丁二必须跟在补丁一之后：两者触碰相同的枚举断言与分发代码块。只用补丁一也可以（跳过第二行）。

配置 / 编译 / 测试见 **[docs/building_zh.md](docs/building_zh.md)**（含 Vulkan SDK、CUDA 工具链、Windows 生成器选择等注意事项），或直接跑 `scripts/build-and-test.sh` / `build-and-test.ps1`。

性能数据见 **[docs/benchmarks_zh.md](docs/benchmarks_zh.md)**（补丁一要点：`IM2COL_FAST_1D` 在大帧长下 CPU 提速 1.03–1.15×；支持分组的 convT 新 kernel 在 Vulkan 上比 legacy im2col 路径快 2.15×。补丁二要点：depthwise-1d 融合版 CPU 提速 19–30×；snake CPU 2.2–4.6× / Vulkan 最高 3.9×；affine_prelu Vulkan 最高 6.3×；gru 补上 RNN 空缺，CPU H512×B4×L32 约 47 ms。）

## 后端支持矩阵

补丁一：

| 算子 | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| `IM2COL_FAST_1D` | ✅ 专用 kernel | ✅ 别名 IM2COL | ✅ 别名 | ✅ 别名 |
| `conv_transpose_1d_ext` | ✅ 全参数 | ✅ `p0=0, d0=1`（分组 ✓） | ✅ 全参数 | ⚠️ 仅 `g0=1, p0=0` |
| `REL_POS_BIAS` | ✅ | ✅ | —（回落 CPU） | — |
| `SCATTER_ELEMENTS` | ✅ | ✅（add 需 `shaderBufferFloat32AtomicAdd`） | — | — |

补丁二：

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

（上游 qvac 中 supertonic 五件套是 Metal 内核、后五件是 Vulkan shader；本移植保留 Vulkan 五件套，Metal 全部先关闭待内核移植，CUDA 亦关闭——上游同样没有 CUDA 实现。）

不支持的参数组合由各后端 `supports_op` 显式拒绝，计算图会干净地回落到 CPU 后端，而不是产出错误结果。

## 许可证与出处

[MIT](LICENSE)，与上游 ggml 一致。各算子的原始实现分属其上游项目（audio.cpp、TTS.cpp、ggmlR、ggml，以及补丁二的 [tetherto/qvac-ext-ggml](https://github.com/tetherto/qvac-ext-ggml)）；本仓库仅做重定基、接口对齐、bug 修复与后端/测试补全。逐算子的致谢与出处见 [docs/operators_zh.md](docs/operators_zh.md)。
