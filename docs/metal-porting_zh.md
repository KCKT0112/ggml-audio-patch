# Metal 移植与验证指南

> 中文 | **[English](metal-porting.md)**

如何把补丁二的十个 qvac 算子带到 Apple Metal 后端、如何验证、以及 Metal 移植达到什么标准才能以 `patches/metal-ops-ggml0190.patch` 的形式合并。

先读 **[/AGENTS.md](../AGENTS.md)**——它定义了所有贡献者（人类或 AI agent）的编辑边界。简版：CPU 内核、测试参考函数、`metal-reference/*` 内核体是**冻结契约**；集成胶水由你编写。

## 当前状态

| 算子 | CPU | Vulkan | Metal |
|---|---|---|---|
| Supertonic × 5 | ✅ | —（回落 CPU） | 📋 有参考代码，**未接线** |
| GRU / ZERO_UPSAMPLE / CHANNEL_SHUFFLE / AFFINE_PRELU | ✅ | ✅ | ❌ 上游无 kernel |
| SNAKE | ✅ | ✅ | 📋 有参考代码，**未接线** |

补丁二**不含 Metal 集成**：`ggml-metal-device.m` 的 `supports_op` 没有这十个算子的 case，`default: return false` 让它们全部干净回落 CPU。macOS 用户当前零收益但也零风险。`metal-reference/` 里是上游已验证过的 kernel 源码和全部 host 侧胶水，随时可以接线。

## 为什么能比较顺利地接上

供体树（qvac）与本仓库的 v0.19.0 基线使用**同一代 Metal 后端架构**，集成关键的名字一一对应（提取时已逐项核对两个树）：

| 项目 | v0.19.0 基线 | qvac 供体 | 是否一致 |
|---|---|---|---|
| kargs typedef 约定（`ggml_metal_kargs_*`，在 `ggml-metal-impl.h`） | ✅ | ✅ | 一致 |
| encoder API（`ggml_metal_encoder_set_buffer` / `_bytes` / `set_pipeline` / `dispatch_threadgroups` / `set_threadgroup_memory_size`） | ✅ `ggml-metal-device.h` | ✅ | 签名一致 |
| pipeline 查找约定（`ggml_metal_library_get_pipeline_*` 在 `ggml-metal-device.cpp`，声明在 `ggml-metal-device.h`） | ✅ | ✅ | 一致 |
| op 分发 switch（`ggml_metal_op_*` 在 `ggml-metal-ops.cpp`，声明在 `ggml-metal-ops.h`） | ✅ | ✅ | 一致 |
| `ggml-metal.metal` 里的 `erf_approx<T>` 模板 + `SQRT_2_INV` 常量 | ✅（`kernel_gelu_erf_f32` 在用） | ✅ | 一致 |

除三处"追加"位置外，需要改动的只有 op 分发 switch 和 supports_op switch 两个 case 区。

## 前置条件（验证的唯一途径）

- macOS 13+，Apple Silicon（或 AMD GPU），Xcode 15+ 工具链。
- 已打补丁的源码树：v0.19.0 + 补丁一 + 补丁二。
- 先有绿色的 CPU 基线：`cmake -B build-metal -DGGML_METAL=ON ...` 配置构建，`test_qvac_ops_cpu cpu` 全过——Metal 工作从绿色基线开始。

## 集成步骤（一次一个算子，每步独立可验证）

推荐顺序：

1. `supertonic_bias_gelu`（最简单，有现成的图内对照：`ggml_add` + `ggml_gelu_erf`）
2. `supertonic_pw2_residual`（纯逐元素）
3. `supertonic_edge_pad_1d`
4. `snake`
5. `supertonic_layer_norm_channel`（simdgroup 归约）
6. `supertonic_depthwise_1d`（三种 kernel 宽度展开 + causal 标志）

每个算子：

1. **Kernel**：把 `metal-reference/supertonic_ops.metal` 里对应 kernel 追加进 `src/ggml-metal/ggml-metal.metal`（内核体不许改——AGENTS.md 规则 3）。
2. **kargs**：把 `metal-reference/host-side.cpp`（PIECE 1）的 struct 拷进 `src/ggml-metal/ggml-metal-impl.h`。字段顺序是 ABI——规则 4。
3. **Pipeline 查找**：把 `ggml_metal_library_get_pipeline_*` 函数（PIECE 3，`#if 0` 块内）拷进 `ggml-metal-device.cpp`，并在 `ggml-metal-device.h` 按既有风格加声明。
4. **分发函数**：把 `ggml_metal_op_*`（PIECE 4）拷进 `ggml-metal-ops.cpp`，在 `ggml-metal-ops.h` 加声明，并接上 switch case（`case GGML_OP_SUPERTONIC_BIAS_GELU: { n_fuse = ggml_metal_op_supertonic_bias_gelu(ctx, idx); } break;`）。
5. **门控**：在 `ggml-metal-device.m` 的 `supports_op` 里加这一个算子的 case（从 PIECE 5 抄）——注意只开这一个。
6. **构建**：重新编译。基线在构建期把 `ggml-metal.metal` 编译成 metallib，shader 语法错误会在这一步暴露。
7. **测试**（见下），通过后再做下一个算子。

## 测试挂具（补丁二已接好钩子）

`tests/test_qvac_ops.c` 接受 `metal` 后端参数。编译 Metal 变体（仅 macOS）：

```bash
clang -O2 -DUSE_METAL -I ggml-src/include -I ggml-src/src \
  -o test_qvac_ops_metal tests/test_qvac_ops.c \
  -L ggml-src/build-metal/src -lggml-base -lggml-cpu -lggml-metal \
  -framework Foundation -framework Metal -framework MetalKit
./test_qvac_ops_metal metal
```

门控全关（当前出货状态）时，每个 case 打印 `SKIP: metal does not support this op shape` 并以 `ALL PASSED` 结束——这是接线前的正确状态，也可当作回落路径的体检。

## 验证流程（每个算子开完门控后跑）

三个都跑——Metal 的结果只在两个绿色基线之间才有意义：

```bash
./test_qvac_ops_metal cpu      # CPU 回归（必须保持 ALL PASSED）
./test_qvac_ops_cpu   cpu      # 如分开构建
./test_qvac_ops_metal metal    # 新路径
```

Metal 上的正确性无需额外写对照：测试内部已内置手写 CPU 参考，Metal op 的输出直接与之比对。

## 验收标准（全部强制）

1. **正确性**：`test_qvac_ops metal` 报 `ALL PASSED`，且启用算子的 case 矩阵完整跑过（depthwise 5 例含 causal K=7 与 dilation；LN 3；pw2 2；bias_gelu 2；edge_pad 4；snake 2）。容差保持出货值（逐元素 1e-3；bias_gelu/snake 的 erf 路径 1e-4）。
2. **无 CPU 回归**：同一批二进制下 `test_qvac_ops cpu` 与 `test_learned_ops cpu` 仍然 `ALL PASSED`。
3. **干净回落**：未启用的形状（如 K=9 的 depthwise，或没有 Metal kernel 的四个算子）`ggml_backend_supports_op` 必须返回 false，且图经回落路径产出与 CPU 一致的结果——每个算子给一个故意不支持的 case 做演示（harness 打印 SKIP；在 PR 里附说明）。
4. **门控纪律**：`supports_op` 只对通过第 1 条验收的精确（类型、形状、参数）包络返回 true——不得是超集。
5. **Kernel diff 纪律**：你的分支里 `git diff metal-reference/` 必须为空（内核零改动；AGENTS.md 规则 3/4）。全部改动只落在四个集成文件 + supports_op 门控。
6. **确定性**：Metal 套件连跑两次，结果完全一致。
7. **平台声明**：PR 描述写明 OS 版本、芯片/GPU、Xcode 版本，以及 `test_qvac_ops_metal metal` 输出结尾原文（`ALL PASSED` 块）。
8. **基准（鼓励、不强制）**：仿照现有写法给 `tests/bench_qvac_ops.c` 加 `metal` 分支，按文档表格格式汇报 融合 vs 组合 数据（含设备/OS）。`docs/benchmarks*.md` 只收真实运行的数据。

## 已知坑（来自供体树）

- **kargs 是按位置绑定的 ABI**：Metal 按字段顺序绑定 `constant & args`。host struct 与 kernel struct 必须逐字段一致——不许重排、不许中间插字段（尾部追加需按 AGENTS.md 规则 4 双侧协调）。
- **`layer_norm_channel` 的 threadgroup**：`nth` 必须是 32（simdgroup）的倍数且 ≤ 256；参考分发函数已经这么算了。共享内存 `8 * sizeof(float)`——每个 simdgroup 一个 float。
- **`depthwise_1d` 的展开是编译期的**（K ∈ {3, 5, 7}）；其他 K 会落进按 K=3 处理的 else 分支——supports_op 门控必须拒绝其他 K（上游就是这么门的，照做）。
- **`bias_gelu` 位兼容**：kernel 用基线的 `erf_approx`（Abramowitz–Stegun/Hastings 多项式，与基线 `kernel_gelu_erf_f32` 同款），而 CPU 参考用 `erff`。测试的 1e-4 容差吸收了多项式差异；**不要**把 kernel "修"成精确 `erff`——那会破坏与未融合 Metal gelu 路径的位一致性。
- **`snake` 的 grid-stride**：分发端把 threadgroup 数封顶在 65535，更大的 N 靠 grid-stride 循环覆盖——别"简化"成一线程一元素的分发。
- **占位 buffer 绑定**：`depthwise_1d` 在 `bias == NULL` 时把 `src[0]` 绑到 index 3 当占位（Metal 要求声明的 buffer 全部绑定）。保留这个行为。
- **Vulkan 侧的经验同样适用**（见 `docs/porting-notes_zh.md`）：每个图变体独立分配器、不走 sched 等。

## 不要做的事

- 不要一次性开全部门控、六个 kernel 一起调。
- 不要改 `tests/test_qvac_ops.c` 的参考函数（AGENTS.md 规则 2）、`metal-reference/*`（规则 3/4）或直接改补丁二。
- 不要提交本地路径（`/Users/...`）或不可复现的论断。

分支满足全部验收清单后，按标准再生成流程（AGENTS.md 产物表）把集成固化成 `patches/metal-ops-ggml0190.patch`（叠加在补丁二之上）。
