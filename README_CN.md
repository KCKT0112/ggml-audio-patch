# ggml-audio-patch

> 中文 | **[English](README.md)**

一套精选补丁，把**十六个音频域算子**统一移植进 [ggml](https://github.com/ggml-org/ggml) **v0.19.0**：十四个取自 ggml 生态不同项目——API 对齐上游规范、修复原生实现的 bug、并按后端能力补齐 CPU / Vulkan / CUDA / Metal 支持；另有两个为 NSF-HiFiGAN 声码器在本仓库新写（融合 bias 加 + leaky ReLU；带生产侧融合的 stride-1 直接 1D 卷积）。以四个统一 diff 交付（按序应用，Metal 补丁可按平台跳过），另附正确性测试与跨后端基准程序。

## 补丁一：六个 learned 算子

| 算子 | 学习来源 | 价值 |
|---|---|---|
| `GGML_OP_IM2COL_FAST_1D` | [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) | 1D im2col 的每行有效窗口 O(1) 直接算出，不再扫完整核宽；`d0 == 1` 时窗口中段退化为纯 `memcpy`。 |
| `ggml_conv_transpose_1d_ext` | [mmwillet/TTS.cpp](https://github.com/mmwillet/TTS.cpp)（ggml `support-for-tts` 分支） | 与 PyTorch `ConvTranspose1d` 全参数对齐：**groups / output_padding / padding**；并修复其损坏的 CPU 分组路径、CUDA `op_params` 读错位与除零断言。 |
| `GGML_OP_REL_POS_BIAS` | [ggmlR](https://CRAN.R-project.org/package=ggmlR)（ggml 的 R 绑定） | BoTNet 风格双轴相对位置注意力偏置：按轴查位移表 + 逐通道点积，含 CPU 与 Vulkan 实现。 |
| `GGML_OP_SCATTER_ELEMENTS` | [ggmlR](https://CRAN.R-project.org/package=ggmlR) | ONNX `ScatterElements` 语义——`get_rows` 的逆操作。Vulkan 端用 `VK_EXT_shader_atomic_float` 原子加实现累加归约。 |
| `GGML_OP_ADD_LEAKY_RELU` | 为 [pc-nsf-hifigan.cpp](https://github.com/KakaruHayate/pc-nsf-hifigan.cpp)（NSF-HiFiGAN 声码器）新写 | `y = leaky(a + b)` 单次遍历——广播 `[1,C]` 或逐行 `[T,C]` bias；与 `add` + `leaky_relu` 组合逐位一致。同时把原生 CPU `leaky_relu` 内核并行化（上游强制 `n_tasks = 1`）。 |
| `GGML_OP_CONV_DIRECT_1D`（含 `_fused`） | 为 pc-nsf-hifigan.cpp 新写 | stride-1 直接 1D 卷积、无 im2col scratch：权重每次调用打包成 `[IC·K, OCp]`，AVX2 6×16 微内核 + 指针递增寻址（单线程比 `imul` 索引快 26%）。`_fused` 把 bias / 残差 / 输出 leaky 与输入侧 `leaky·scale` **逐位一致**地折入——全声码器 2.85×（80.0 s → 28.0 s，24 线程，16C/32T）。 |

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

## 补丁三：已验证的 Metal 集成

**叠加在补丁二之上**。它把五个 Supertonic kernel 和直接 `GGML_OP_SNAKE` 分发接入 Metal，并用严格的 F32/形状/参数门控限定支持范围。`GRU` / `ZERO_UPSAMPLE` / `CHANNEL_SHUFFLE` / `AFFINE_PRELU` 因供体没有 Metal kernel，仍干净回落 CPU。验证矩阵和平台证据见 [docs/metal-porting_zh.md](docs/metal-porting_zh.md)。

`GRU` / `ZERO_UPSAMPLE` / `CHANNEL_SHUFFLE` / `AFFINE_PRELU` / `SNAKE` 五个另有 Vulkan compute shader 实现，`GRU` 额外带 H = 2/4/8 的寄存器驻留变体，共享内存上限 H ≤ 128。

## 补丁四：`CONV_DIRECT_1D` 的 Vulkan compute 后端

**叠加在补丁一、二之上**（补丁二占用了相同的 shader 注册表与分发插入点，因此必须按 1 → 2 → 4 顺序应用；补丁三仅 Metal、与其正交——装不装它，补丁四同样应用）。这是补丁一声码器工作的 GPU 半边：不改变任何算子语义——`tests/test_learned_ops.c` 仍是冻结契约。

- `vulkan-shaders/conv_direct_1d.comp` —— 隐式 GEMM 的 stride-1 直接 1D 卷积，仅 fp32。单一 SPIR-V；warp 分块以 Vulkan specialization constants 传入（`mul_mm` 方案），生成五条具体 pipeline。`ggml_vk_op_get_pipeline` 按输出通道数逐节点选择分块（OC ≤ 16 → e16，≤ 32 → e32，否则 w128）；**任意变体对任意 shape 数值一致**，选择纯属性能启发式，不含任何厂商特定假设。
- bias、残差、输出侧 leaky、输入侧 `leaky·scale` 融合与 CPU `_fused` 算子契约完全一致（同一 op_params 布局）。
- `supports_op` 门控：fp32 + 连续内存，`K ≥ 3` 且 `(K−1)·dilation ≤ 72`（共享内存 halo 上限）；不满足返回 false，调度器自动回落 CPU 内核。**以 raw graph 模式直接驱动 ggml-vulkan 的消费方**（直接调 `ggml_vk_build_graph`、绕过调度器的 `supports_op` 检查）必须自行执行同一门控——NSF-HiFiGAN 中 K = 2 的亚像素上采样卷积超出该 shader 的暂存行窗口，不加门控会静默产生错误输出。pc-nsf-hifigan.cpp 用 `PCNSF_DIRECT_MIN_K` 实现（Vulkan 默认 3，其余后端 1）。

全 NSF-HiFiGAN 声码器实测（RTX 2070，32.0.12.x 级驱动，全程 fp32，`GGML_VK_DISABLE_COOPMAT2=1` 关闭张量核；pc-nsf-hifigan.cpp `hifigan_cli` 参考音频多次运行取中位）：**端到端约 433–479 ms**，对比同机 ONNX Runtime DML EP 326 ms / CPU EP 5 155 ms。对 torch-CPU 参考的精度：corr 0.99999985、max|Δ| 6.1e-4——比同片 ORT-DML 紧一个数量级（corr 0.99999697、max|Δ| 2.1e-3）。完整数据与复现命令见 [docs/benchmarks_zh.md](docs/benchmarks_zh.md#补丁四vulkan-conv_direct_1d)。

基线：ggml [`30bf868`](https://github.com/ggml-org/ggml)（v0.19.0）。diff 只在枚举/builder/kernel 的插入点上做增量，应用到邻近 commit 通常只需少量冲突处理。

## 目录结构

```
ggml-audio-patch/
├── patches/
│   ├── learned-ops-ggml0190.patch   # 基于 ggml v0.19.0 的统一 diff（补丁一）
│   ├── qvac-ops-ggml0190.patch      # 叠加在补丁一之上的统一 diff（补丁二）
│   ├── metal-ops-ggml0190.patch     # 叠加在补丁二之上的 Metal 集成（补丁三）
│   └── vulkan-conv-direct-1d-ggml0190.patch  # CONV_DIRECT_1D 的 Vulkan 后端（补丁四，叠加于一+二）
├── tests/
│   ├── test_learned_ops.c           # 补丁一正确性冒烟测试（手写参考值对照）
│   ├── test_qvac_ops.c              # 补丁二正确性测试（cpu | vk | metal 挂具钩子）
│   ├── bench_learned_ops.c          # 补丁一 CPU / Vulkan / CUDA 微基准
│   └── bench_qvac_ops.c             # 补丁二 融合 vs 组合图 基准（CPU + Vulkan + Metal）
├── scripts/
│   ├── build-and-test.sh            # Linux/macOS 一键构建+测试
│   └── build-and-test.ps1           # Windows (pwsh) 一键构建+测试
├── AGENTS.md                        # 贡献与编辑边界（人类或 AI agent 通用）
└── docs/
    ├── building.md / building_zh.md            # 构建须知
    ├── benchmarks.md / benchmarks_zh.md        # 性能测试与方法论
    ├── operators.md / operators_zh.md          # 各算子设计笔记与 API
    ├── metal-porting.md / metal-porting_zh.md  # Metal 集成笔记：状态、挂具、坑、已验证交付
    ├── porting-notes.md / porting-notes_zh.md  # 移植坑与修复记录
    └── task-package.md / task-package_zh.md    # 平台绑定任务如何打包下发
```

## 快速开始

```bash
git clone https://github.com/ggml-org/ggml.git ggml-src
cd ggml-src && git checkout 30bf868        # v0.19.0
git apply ../ggml-audio-patch/patches/learned-ops-ggml0190.patch            # 补丁一
git apply ../ggml-audio-patch/patches/qvac-ops-ggml0190.patch               # 补丁二（顺序应用）
git apply ../ggml-audio-patch/patches/metal-ops-ggml0190.patch              # 补丁三（Metal，可选）
git apply ../ggml-audio-patch/patches/vulkan-conv-direct-1d-ggml0190.patch  # 补丁四（叠加于一+二）
```

补丁二必须跟在补丁一之后：两者触碰相同的枚举断言与分发代码块。非 Metal 平台可不应用补丁三。补丁四必须跟在补丁一、二之后：它消费补丁一的 `CONV_DIRECT_1D`，并与补丁二共用 shader 注册表/分发插入点（与补丁三正交——无论装不装 Metal 补丁都可同样应用）。只装补丁一可以（跳过后续）；只装一+四**不支持**。

配置 / 编译 / 测试见 **[docs/building_zh.md](docs/building_zh.md)**（含 Vulkan SDK、CUDA 工具链、Windows 生成器选择等注意事项），或直接跑 `scripts/build-and-test.sh` / `build-and-test.ps1`。

性能数据见 **[docs/benchmarks_zh.md](docs/benchmarks_zh.md)**（补丁一要点：`IM2COL_FAST_1D` 在大帧长下 CPU 提速 1.03–1.15×；支持分组的 convT 新 kernel 在 Vulkan 上比 legacy im2col 路径快 2.15×；`CONV_DIRECT_1D`+融合在全 NSF-HiFiGAN 声码器上 2.85×（80 002 → 28 030 ms，16C/32T Xeon E5-2675 v3、24 线程；仍比 ONNX Runtime CPU EP 慢约 5.4×，差距如实写入文档）。补丁二/三要点：depthwise-1d 融合版 CPU 提速 19–30×、Apple M4 Metal 最高 5.80×；Snake 在 Metal 达 3.05×、Vulkan 最高 3.9×；Metal 通道 layer norm 达 3.24×；affine_prelu Vulkan 最高 6.3×；gru 补上 RNN 空缺，CPU H512×B4×L32 约 47 ms。补丁四要点：同一直接卷积的 Vulkan 后端以 fp32 跑完整声码器约 430–480 ms（对 ORT DML EP 326 ms），fp32 一致性比 DML 紧一个数量级（max|Δ| 6.1e-4 对 2.1e-3）。）

## 后端支持矩阵

补丁一：

| 算子 | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| `IM2COL_FAST_1D` | ✅ 专用 kernel | ✅ 别名 IM2COL | ✅ 别名 | ✅ 别名 |
| `conv_transpose_1d_ext` | ✅ 全参数 | ✅ `p0=0, d0=1`（分组 ✓） | ✅ 全参数 | ⚠️ 仅 `g0=1, p0=0` |
| `REL_POS_BIAS` | ✅ | ✅ | —（回落 CPU） | — |
| `SCATTER_ELEMENTS` | ✅ | ✅（add 需 `shaderBufferFloat32AtomicAdd`） | — | — |
| `ADD_LEAKY_RELU` | ✅ | —（回落 CPU） | — | — |
| `CONV_DIRECT_1D`（含 `_fused`） | ✅ | ✅ 补丁四（fp32，`K ≥ 3`，`(K−1)·dil ≤ 72`，否则回落 CPU） | — | — |

补丁二：

| 算子 | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| `SUPERTONIC_DEPTHWISE_1D`（含 `_ct` / `_causal_ct`） | ✅ | —（回落 CPU） | — | ✅ F32，K ∈ {3,5,7} |
| `SUPERTONIC_LAYER_NORM_CHANNEL`（含 `_ct`） | ✅ | — | — | ✅ F32 |
| `SUPERTONIC_PW2_RESIDUAL`（含 `_ct`） | ✅ | — | — | ✅ F32 |
| `SUPERTONIC_BIAS_GELU`（含 `_ct`） | ✅ | — | — | ✅ F32 |
| `SUPERTONIC_EDGE_PAD_1D`（含 `_ct`） | ✅ | — | — | ✅ F32 |
| `GRU` | ✅ | ✅ H ≤ 128（含 H=2/4/8 变体） | — | — |
| `ZERO_UPSAMPLE` | ✅ | ✅ | — | — |
| `CHANNEL_SHUFFLE` | ✅ | ✅ | — | — |
| `AFFINE_PRELU` | ✅ | ✅ | — | — |
| `SNAKE` | ✅ | ✅ | — | ✅ F32 |

（补丁三接入供体的五个 Supertonic Metal kernel 和直接 Snake 分发；另外四个 qvac 算子没有供体 Metal kernel，仍显式关闭。集成笔记与验证证据见 [docs/metal-porting_zh.md](docs/metal-porting_zh.md)。CUDA 仍关闭，与供体一致。贡献与编辑边界：[AGENTS.md](AGENTS.md)。本轮无本地苹果硬件如何打包下发任务：[docs/task-package_zh.md](docs/task-package_zh.md)。）

不支持的参数组合由各后端 `supports_op` 显式拒绝，计算图会干净地回落到 CPU 后端，而不是产出错误结果。

## 许可证与出处

[MIT](LICENSE)，与上游 ggml 一致。各算子的原始实现分属其上游项目（audio.cpp、TTS.cpp、ggmlR、ggml，以及补丁二的 [tetherto/qvac-ext-ggml](https://github.com/tetherto/qvac-ext-ggml)）；本仓库仅做重定基、接口对齐、bug 修复与后端/测试补全。逐算子的致谢与出处见 [docs/operators_zh.md](docs/operators_zh.md)。
