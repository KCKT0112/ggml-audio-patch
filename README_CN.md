# ggml-audio-patch

> 中文 | **[English](README.md)**

一套精选补丁，把来自 ggml 生态不同项目的**四个音频域算子**统一移植进 [ggml](https://github.com/ggml-org/ggml) **v0.19.0**：API 对齐上游规范、修复了原生实现的 bug、并按后端能力补齐 CPU / Vulkan / CUDA 支持。以单个统一 diff + 正确性测试 + 跨后端基准程序交付。

## 四个算子

| 算子 | 学习来源 | 价值 |
|---|---|---|
| `GGML_OP_IM2COL_FAST_1D` | [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) | 1D im2col 的每行有效窗口 O(1) 直接算出，不再扫完整核宽；`d0 == 1` 时窗口中段退化为纯 `memcpy`。 |
| `ggml_conv_transpose_1d_ext` | [mmwillet/TTS.cpp](https://github.com/mmwillet/TTS.cpp)（ggml `support-for-tts` 分支） | 与 PyTorch `ConvTranspose1d` 全参数对齐：**groups / output_padding / padding**；并修复其损坏的 CPU 分组路径、CUDA `op_params` 读错位与除零断言。 |
| `GGML_OP_REL_POS_BIAS` | [ggmlR](https://CRAN.R-project.org/package=ggmlR)（ggml 的 R 绑定） | BoTNet 风格双轴相对位置注意力偏置：按轴查位移表 + 逐通道点积，含 CPU 与 Vulkan 实现。 |
| `GGML_OP_SCATTER_ELEMENTS` | [ggmlR](https://CRAN.R-project.org/package=ggmlR) | ONNX `ScatterElements` 语义——`get_rows` 的逆操作。Vulkan 端用 `VK_EXT_shader_atomic_float` 原子加实现累加归约。 |

基线：ggml [`30bf868`](https://github.com/ggml-org/ggml)（v0.19.0）。diff 只在枚举/builder/kernel 的插入点上做增量，应用到邻近 commit 通常只需少量冲突处理。

## 目录结构

```
ggml-audio-patch/
├── patches/
│   └── learned-ops-ggml0190.patch   # 基于 ggml v0.19.0 的统一 diff
├── tests/
│   ├── test_learned_ops.c           # 正确性冒烟测试（手写参考值对照）
│   └── bench_learned_ops.c          # CPU / Vulkan / CUDA 微基准
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
git apply ../ggml-audio-patch/patches/learned-ops-ggml0190.patch
```

配置 / 编译 / 测试见 **[docs/building_zh.md](docs/building_zh.md)**（含 Vulkan SDK、CUDA 工具链、Windows 生成器选择等注意事项），或直接跑 `scripts/build-and-test.sh` / `build-and-test.ps1`。

性能数据见 **[docs/benchmarks_zh.md](docs/benchmarks_zh.md)**（要点：`IM2COL_FAST_1D` 在大帧长下 CPU 提速 1.03–1.15×；支持分组的 convT 新 kernel 在 Vulkan 上比 legacy im2col 路径快 2.15×）。

## 后端支持矩阵

| 算子 | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| `IM2COL_FAST_1D` | ✅ 专用 kernel | ✅ 别名 IM2COL | ✅ 别名 | ✅ 别名 |
| `conv_transpose_1d_ext` | ✅ 全参数 | ✅ `p0=0, d0=1`（分组 ✓） | ✅ 全参数 | ⚠️ 仅 `g0=1, p0=0` |
| `REL_POS_BIAS` | ✅ | ✅ | —（回落 CPU） | — |
| `SCATTER_ELEMENTS` | ✅ | ✅（add 需 `shaderBufferFloat32AtomicAdd`） | — | — |

不支持的参数组合由各后端 `supports_op` 显式拒绝，计算图会干净地回落到 CPU 后端，而不是产出错误结果。

## 许可证与出处

[MIT](LICENSE)，与上游 ggml 一致。各算子的原始实现分属其上游项目（audio.cpp、TTS.cpp、ggmlR、ggml）；本仓库仅做重定基、接口对齐、bug 修复与后端/测试补全。逐算子的致谢与出处见 [docs/operators_zh.md](docs/operators_zh.md)。
