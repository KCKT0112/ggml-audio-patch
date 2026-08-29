# Metal 集成笔记

> 中文 | **[English](metal-porting.md)**

可选补丁三（`patches/metal-ops-ggml0190.patch`）接入了 Metal 后端的哪些内容、测试挂具怎么跑、维护时需要知道的坑，以及已验证交付。贡献者（人类或 agent）的编辑边界见 [/AGENTS.md](../AGENTS.md)。本轮如何在没有本地苹果硬件的情况下打包并下发这项任务，记录在 [task-package_zh.md](task-package_zh.md)。

## 当前状态

| 算子 | CPU | Vulkan | Metal |
|---|---|---|---|
| Supertonic × 5 | ✅ | —（回落 CPU） | ✅ 补丁三，F32 |
| GRU / ZERO_UPSAMPLE / CHANNEL_SHUFFLE / AFFINE_PRELU | ✅ | ✅ | ❌ 上游无 kernel |
| SNAKE | ✅ | ✅ | ✅ 补丁三，F32 |

补丁二刻意**不含 Metal 集成**。可选的顺序补丁三接入供体（qvac-ext-ggml，MIT；逐算子致谢见 [operators_zh.md](operators_zh.md)）实际存在的六个 kernel，其余四个算子仍干净回落 CPU。

补丁三还恢复了基线 Metal 的 `GGML_OP_REPEAT` 门控；补丁一曾误将它与更严格的分组/带 padding `CONV_TRANSPOSE_1D` 门控绑定。纯 Metal 组合图基线依赖这一修正。

## 测试挂具

`tests/test_qvac_ops.c` 接受 `metal` 后端参数。编译 Metal 变体（仅 macOS），Metal 运行夹在两个绿色 CPU 基线之间：

```bash
clang -O2 -DUSE_METAL -I ggml-src/include -I ggml-src/src \
  -o test_qvac_ops_metal tests/test_qvac_ops.c \
  -L ggml-src/build-metal/src -L ggml-src/build-metal/src/ggml-metal \
  -lggml-base -lggml-cpu -lggml-metal \
  -framework Foundation -framework Metal -framework MetalKit

./test_qvac_ops_metal cpu      # CPU 回归（必须保持 ALL PASSED）
./test_qvac_ops_cpu   cpu      # 如分开构建
./test_qvac_ops_metal metal    # Metal 路径
```

应用补丁三后，18 个 Supertonic/Snake case 在 Metal 实际执行；GRU/zero-upsample/channel-shuffle/affine-PReLU 的 16 个 case 继续打印预期的干净回落 `SKIP`。Metal 上的正确性无需额外写对照：测试内部已内置手写 CPU 参考，Metal op 的输出直接与之比对。交付级验证请把 Metal 套件连跑两遍并要求结果完全一致（见下文[已验证交付](#已验证交付)）。

## 已知坑

- **kargs 是按位置绑定的 ABI**：Metal 按字段顺序绑定 `constant & args`。host struct 与 kernel struct 必须逐字段一致——不许重排、不许中间插字段（尾部追加是 host 与 shader 两侧的协调变更）。
- **`layer_norm_channel` 的 threadgroup**：`nth` 必须是 32（simdgroup）的倍数且 ≤ 256；分发函数已经这么算了。共享内存 `8 * sizeof(float)`——每个 simdgroup 一个 float。集成 kernel 在所有 simdgroup 读取归约后的 mean 之后、variance 部分复用 `shared[0]` 之前增加一道 barrier；供体漏了这道同步。C=256/L=4096 压力实验中，未修补版本 10 个独立进程有 9 个结果错误，加入 barrier 后 20/20 通过。
- **`depthwise_1d` 的展开是编译期的**（K ∈ {3, 5, 7}）；其他 K 会落进按 K=3 处理的 else 分支——supports_op 门控拒绝其他 K 值，与上游一致。
- **`bias_gelu` 位兼容**：kernel 用基线的 `erf_approx`（Abramowitz–Stegun/Hastings 多项式，与基线 `kernel_gelu_erf_f32` 同款），而 CPU 参考用 `erff`。测试的 1e-4 容差吸收了多项式差异；**不要**把 kernel "修"成精确 `erff`——那会破坏与未融合 Metal gelu 路径的位一致性。
- **`snake` 与基线重叠**：v0.19.0 已含供组合图融合使用的泛型 `kernel_snake`。补丁三为直接 `GGML_OP_SNAKE` 分发复用其相同公式和 pipeline，避免重复定义 `kernel_snake_f32` 符号。
- **占位 buffer 绑定**：`depthwise_1d` 在 `bias == NULL` 时把 `src[0]` 绑到 index 3 当占位（Metal 要求声明的 buffer 全部绑定）。保留这个行为。
- **Vulkan 侧的经验同样适用**（见 [porting-notes_zh.md](porting-notes_zh.md)）：每个图变体独立分配器、不走 sched 等。

## 已验证交付

2026-08-29 在 macOS 27.0（26A5421a）、Apple M4（10 核 GPU）、Xcode 27.0（27A5237l）、Apple Clang 21.0.0 上验证：

- 补丁一 → 补丁二 → 补丁三可干净应用到原始 `30bf868`；
- Metal 构建成功，18 个已启用 Metal case 全部实际执行且没有 SKIP；
- 无 Metal kernel 的四个算子共 16 个 case 全部干净 SKIP；
- 显式门控检查拒绝 depthwise K=9、非法 layout、Snake 混合类型及四个无 kernel 算子；
- `test_qvac_ops_metal cpu`、`test_qvac_ops_cpu cpu`、`test_learned_ops_cpu cpu` 均为 `ALL PASSED`；
- 两次 Metal 运行的测试结果投影完全一致，结尾均为：

```text
[test] metal supports_op gates / fallback envelope
  done (0 failures so far)

ALL PASSED
```
