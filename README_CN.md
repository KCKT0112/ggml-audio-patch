# ggml-audio-patch

> 中文 | **[English](README.md)**

一套精选补丁，把**十六个音频域算子**统一移植进 [ggml](https://github.com/ggml-org/ggml) **v0.19.0**：十四个取自 ggml 生态不同项目——API 对齐上游规范、修复原生实现的 bug、并按后端能力补齐 CPU / Vulkan / CUDA / Metal 支持；另有两个为 NSF-HiFiGAN 声码器在本仓库新写（融合 bias 加 + leaky ReLU；带生产侧融合的 stride-1 直接 1D 卷积）。以八个统一 diff 交付（按序应用，Metal 补丁三/六可按平台跳过；补丁五为 Vulkan 管线缓存增强），另附正确性测试与跨后端基准程序。

## 补丁一：六个 learned 算子

| 算子 | 学习来源 | 价值 |
|---|---|---|
| `GGML_OP_IM2COL_FAST_1D` | [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) | 1D im2col 的每行有效窗口 O(1) 直接算出，不再扫完整核宽；`d0 == 1` 时窗口中段退化为纯 `memcpy`。 |
| `ggml_conv_transpose_1d_ext` | [mmwillet/TTS.cpp](https://github.com/mmwillet/TTS.cpp)（ggml `support-for-tts` 分支） | 与 PyTorch `ConvTranspose1d` 全参数对齐：**groups / output_padding / padding**；并修复其损坏的 CPU 分组路径、CUDA `op_params` 读错位与除零断言。 |
| `GGML_OP_REL_POS_BIAS` | [ggmlR](https://CRAN.R-project.org/package=ggmlR)（ggml 的 R 绑定） | BoTNet 风格双轴相对位置注意力偏置：按轴查位移表 + 逐通道点积，含 CPU 与 Vulkan 实现。 |
| `GGML_OP_SCATTER_ELEMENTS` | [ggmlR](https://CRAN.R-project.org/package=ggmlR) | ONNX `ScatterElements` 语义——`get_rows` 的逆操作。Vulkan 端用 `VK_EXT_shader_atomic_float` 原子加实现累加归约。 |
| `GGML_OP_ADD_LEAKY_RELU` | 为 [pc-nsf-hifigan.cpp](https://github.com/KakaruHayate/pc-nsf-hifigan.cpp)（NSF-HiFiGAN 声码器）新写 | `y = leaky(a + b)` 单次遍历——广播 `[1,C]` 或逐行 `[T,C]` bias；与 `add` + `leaky_relu` 组合逐位一致。同时把原生 CPU `leaky_relu` 内核并行化（上游强制 `n_tasks = 1`）。 |
| `GGML_OP_CONV_DIRECT_1D`（含 `_fused`） | 为 pc-nsf-hifigan.cpp 新写 | stride-1 直接 1D 卷积、无 im2col scratch：权重每次调用打包成 `[IC·K, OCp]`，AVX2 6×16 微内核 + 指针递增寻址（单线程比 `imul` 索引快 26%）。`_fused` 把 bias / 残差 / 输出 leaky 与输入侧 `leaky·scale` **逐位一致**地折入——全声码器 2.50×（11.09 s → 4.43 s，24 线程，16C/32T），fp32 下与 ONNX Runtime CPU EP（5.155 s）持平。 |

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

## 补丁五：Vulkan 持久化磁盘管线缓存

移植自 [KakaruHayate/game.cpp](https://github.com/KakaruHayate/game.cpp)（`cmake/patches/ggml-vulkan-pipeline-cache`），适配 ggml **v0.19.0**：每进程一个 `VkPipelineCache`，设备初始化时从磁盘载入，喂给每次 `vkCreateComputePipelines`，在管线被创建/销毁后回写磁盘。不含任何算子/shader 内容——数值上按构造零影响。

- 已验证在原生 v0.19.0 上**独立可用**，同时可叠加在补丁 1–4 之上（补丁四、五都改 `src/ggml-vulkan/ggml-vulkan.cpp`；推荐顺序 1 → 2 →（3）→ 4 → 5）。
- 环境开关：`GGML_VK_PIPELINE_CACHE_PATH`（自定义缓存文件位置；默认 `ggml_audio_vk_pipeline.cache`，Windows 下位于 `%LOCALAPPDATA%`，Linux 下位于 `$HOME/.cache`）、`GGML_VK_DISABLE_PIPELINE_CACHE=1`（关闭）、`GGML_VK_PIPELINE_CACHE_DEBUG=1`（打印载入/写回字节数）。
- 实测（2026-08-30，Windows 11，Xeon E5-2675 v3，RTX 2070 驱动 32.0.16.2002，MSVC 14.29，pc-nsf-hifigan.cpp `hifigan_cli`，参考片段 T=1722，blob 334 150 B）：无 blob 时新补丁二进制的首跑会把管线编译计进第一次计算——图内 1 061.5 ms / 进程 wall 2 454 ms；此后带 blob 的进程重启稳定在图内 346.8–355.0 ms / wall ≈1 120–1 290 ms。首跑成本图内 ≈3×、wall ≈2× 缩减。
- 如实说明：本测试机上 NVIDIA 驱动自身的 shader 缓存（…/NVIDIA/GLCache）同样跨进程保存编译产物；该层命中时，blob 在稳态没有可测增益（交叉实测中位数 ON 376.1 对 OFF 373.1 ms，n=3，机器散布 ±10%）。blob 的价值体现在驱动缓存冷/被逐出时——驱动升级/重装或缓存清理后的首跑、缓存容量受限的配置、缓存能力弱的驱动/平台。复现命令与完整运行表见 [docs/benchmarks_zh.md](docs/benchmarks_zh.md#补丁五--vulkan-管线持久缓存)。

## 补丁六：Metal F32 融合直接卷积

新增 `CONV_DIRECT_1D` / `_fused` 的 Metal implicit-GEMM 实现：`16×64`、`32×64`、`64×64` simdgroup-matrix tile 和 `OC < 8` scalar 路径，融合 bias、residual、输入 scale/leaky 和输出 leaky，省去 im2col 中间张量。补丁也补齐 `IM2COL_FAST_1D` 的 Metal 能力声明。

应用顺序为 **1 → 2 → 3 → 6**，或完整 **1 → 2 → 3 → 4 → 5 → 6**。不修改 CPU/Vulkan/CUDA kernel 或公共算子语义。Metal 要求 F32、连续的单序列张量与 Apple7 simdgroup-matrix 能力；直接执行图的调用方必须先查询 `supports_op`，不支持时选择原有图或 scheduler 回落。

实测与复现见 [Metal 直接卷积](docs/metal-direct-conv_zh.md)：2026-08-31 的 Apple M4 声码器测试中，19.992 s 音频最快稳定轮为 0.823–0.836 s；与 ORT CPU 的相近时间窗口对照为 1.221 s 对 3.156 s。2026-09-05 在集合仓库重新构建并验证 CPU/Metal 算子测试。其他 Apple 设备尚未真机验证。

## 补丁七：音频算子正确性修复

`audio-op-fixes-ggml0190.patch` 修复 CPU/Vulkan ScatterElements 索引处理（合法负索引归一化，越界更新在计算地址前丢弃）、允许与 stride 无整除关系的非负转置卷积 padding，并让 CPU 直接卷积按权重/bias 的字节 stride 取值。原有连续张量、合法正索引的语义不变。在现有补丁栈之后应用，一至六保持不变。`tests/test_audio_op_regressions.cpp` 覆盖 14 项 CPU 回归，也可用 `Vulkan0` 运行支持的用例。详见 [算子修复](docs/audio-op-fixes_zh.md)。

## 补丁八：AVX2 直接卷积工作区对齐

`cpu-direct-conv-alignment-ggml0190.patch` 为打包权重使用非对齐 AVX2 load：CPU graph-plan 工作区来自 `new[]`，不保证 32 字节对齐。它修复了新回归测试暴露的 Linux 崩溃。新增两个测试故意将工作区放在距 32 字节边界偏移 16 字节的位置。在补丁七之后应用，原有补丁文件保持不变。

基线：ggml [`30bf868`](https://github.com/ggml-org/ggml)（v0.19.0）。diff 只在枚举/builder/kernel 的插入点上做增量，应用到邻近 commit 通常只需少量冲突处理。

## 目录结构

```
ggml-audio-patch/
├── patches/
│   ├── learned-ops-ggml0190.patch   # 基于 ggml v0.19.0 的统一 diff（补丁一）
│   ├── qvac-ops-ggml0190.patch      # 叠加在补丁一之上的统一 diff（补丁二）
│   ├── metal-ops-ggml0190.patch     # 叠加在补丁二之上的 Metal 集成（补丁三）
│   ├── vulkan-conv-direct-1d-ggml0190.patch  # CONV_DIRECT_1D 的 Vulkan 后端（补丁四，叠加于一+二）
│   ├── vulkan-pipeline-cache-ggml0190.patch  # Vulkan 持久化磁盘管线缓存（补丁五，可独立或叠加于一至四）
│   └── metal-conv-direct-1d-ggml0190.patch  # Metal F32 direct convolution (6)
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
git apply ../ggml-audio-patch/patches/vulkan-pipeline-cache-ggml0190.patch  # 补丁五（在补丁四之后；也可独立应用于原生 v0.19.0）
git apply ../ggml-audio-patch/patches/metal-conv-direct-1d-ggml0190.patch   # 补丁六（需要一+二+三）
git apply ../ggml-audio-patch/patches/audio-op-fixes-ggml0190.patch       # patch 7
git apply ../ggml-audio-patch/patches/cpu-direct-conv-alignment-ggml0190.patch  # patch 8
```

补丁二必须跟在补丁一之后：两者触碰相同的枚举断言与分发代码块。非 Metal 平台可不应用补丁三。补丁四必须跟在补丁一、二之后：它消费补丁一的 `CONV_DIRECT_1D`，并与补丁二共用 shader 注册表/分发插入点（与补丁三正交——无论装不装 Metal 补丁都可同样应用）。补丁五已验证既可独立应用于原生 v0.19.0，也可叠加在补丁 1–4 之上——组合应用时**在补丁四之后装**（它与补丁四都改 `src/ggml-vulkan/ggml-vulkan.cpp`）。只装补丁一可以（跳过后续）；只装一+四**不支持**。

配置 / 编译 / 测试见 **[docs/building_zh.md](docs/building_zh.md)**（含 Vulkan SDK、CUDA 工具链、Windows 生成器选择等注意事项），或直接跑 `scripts/build-and-test.sh` / `build-and-test.ps1`。

性能数据见 **[docs/benchmarks_zh.md](docs/benchmarks_zh.md)**（补丁一要点：`IM2COL_FAST_1D` 在大帧长下 CPU 提速 1.03–1.15×；支持分组的 convT 新 kernel 在 Vulkan 上比 legacy im2col 路径快 2.15×；`CONV_DIRECT_1D`+融合在全 NSF-HiFiGAN 声码器上 2.50×（11 089 → 4 431 ms，16C/32T Xeon E5-2675 v3、24 线程——快于 2026-08-30 当日实测的 ONNX Runtime CPU EP 4 724 ms fp32）。补丁二/三要点：depthwise-1d 融合版 CPU 提速 19–30×、Apple M4 Metal 最高 5.80×；Snake 在 Metal 达 3.05×、Vulkan 最高 3.9×；Metal 通道 layer norm 达 3.24×；affine_prelu Vulkan 最高 6.3×；gru 补上 RNN 空缺，CPU H512×B4×L32 约 47 ms。补丁四要点：同一直接卷积的 Vulkan 后端以 fp32 跑完整声码器约 430–480 ms（对当日实测 ORT DML EP 282 ms），fp32 一致性比 DML 紧一个数量级（max|Δ| 6.1e-4 对 2.1e-3）。补丁五要点：Vulkan 管线缓存在驱动缓存冷启动时把首跑管线编译成本图内 ≈3×（1 061.5 → ≈350 ms）、进程 wall ≈2×（2 454 → ≈1 200 ms）压缩，数值零影响。**终态说明（2026-08-30）**：追平 Vulkan 与 DML 剩余差距的路径已勘察并刻意止步——唯一可能路线（f16 张量核 GEMM）经 torch 侧模拟精度退化超 DML 锚点（corr 0.99999969 对 DML 0.99999697；max|Δ| 3.9e-3 对 DML 2.1e-3），而保精度的替代方案天花板均仍在 DML 之上；详见 `docs/benchmarks_zh.md` §6。）

## 后端支持矩阵

补丁一：

| 算子 | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| `IM2COL_FAST_1D` | ✅ 专用 kernel | ✅ 别名 IM2COL | ✅ 别名 | ✅ 别名 |
| `conv_transpose_1d_ext` | ✅ 全参数 | ✅ `p0=0, d0=1`（分组 ✓） | ✅ 全参数 | ⚠️ 仅 `g0=1, p0=0` |
| `REL_POS_BIAS` | ✅ | ✅ | —（回落 CPU） | — |
| `SCATTER_ELEMENTS` | ✅ | ✅（add 需 `shaderBufferFloat32AtomicAdd`） | — | — |
| `ADD_LEAKY_RELU` | ✅ | —（回落 CPU） | — | — |
| `CONV_DIRECT_1D`（含 `_fused`） | ✅ | ✅ 补丁四（fp32，`K ≥ 3`，`(K−1)·dil ≤ 72`，否则回落 CPU） | — | ✅ 补丁六（F32，Apple7） |

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

[MPL-2.0](LICENSE) —— Mozilla Public License 2.0；**例外**：`patches/` 目录采用 `MIT OR Apache-2.0`
双许可（见 [patches/LICENSE](patches/LICENSE)），以保证这些补丁仍可被上游 ggml 接纳。

各算子的原始实现分属其上游项目（audio.cpp、TTS.cpp、ggmlR、ggml，以及补丁二的
[tetherto/qvac-ext-ggml](https://github.com/tetherto/qvac-ext-ggml)）；本仓库仅做重定基、接口对齐、
bug 修复与后端/测试补全。这些上游许可**不因本仓库变更而失效**——原作者给出的授权不可撤销，
源自其代码的部分仍可按原始条款使用。

逐算子的致谢与出处见 [docs/operators_zh.md](docs/operators_zh.md)，完整许可与出处记录见
[NOTICE.md](NOTICE.md)。
