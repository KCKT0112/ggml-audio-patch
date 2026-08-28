# 移植笔记：坑与修复记录

> **[English](porting-notes.md)** | 中文

记录把四个算子移植进 ggml v0.19.0 的固定套路、踩过的坑与修复方式。适合想复现、扩展或继续向 ggml main 提交的人阅读。

## 0. 移植一个新 ggml 算子的固定套路

在 ggml 中新增一个算子需要触碰的文件是固定的，按依赖顺序：

1. **`include/ggml.h`** — 枚举加 `GGML_OP_XXX`（必须放在 `GGML_OP_COUNT` 之前）+ 公开 API 声明。
2. **`src/ggml.c`** —
   - `GGML_OP_NAME[]` / `GGML_OP_SYMBOL[]` 两张表 + 表后的 `static_assert(GGML_OP_COUNT == N)`（**有两条**，名称表和符号表后各一条，都要改）；
   - builder 函数（`ggml_set_op_params` 写参数、`result->src[i]` 挂输入）；
   - 反向传播的 `ggml_visit_parents` 切换与 `ignore_src` 列表（若该 op 无梯度）。
3. **`src/ggml-cpu/ops.h` / `ops.cpp`** — kernel + dispatcher。
4. **`src/ggml-cpu/ggml-cpu.c`** — 三个 switch：compute 派发、`n_tasks`（线程数）、wsize（若有 work buffer）。
5. **各后端** `supports_op` + compute/alias：
   - CUDA：`ggml-cuda.cu` 两处（dispatch + supports_op）+ kernel 文件；
   - Vulkan：单体 `ggml-vulkan.cpp` 十余处（见 §4）+ `.comp` shader + `vulkan-shaders-gen.cpp` 注册；
   - Metal：`ggml-metal-ops.cpp`（dispatch）、`ggml-metal-device.m`（supports_op）、`ggml-metal-device.cpp`（pipeline 断言）；
   - 图拆分器（`ggml-backend-meta.cpp` 一类）也要加 case，否则带新算子的图进 meta 后端会被漏掉。

## 1. IM2COL_FAST_1D：别名式移植 + 一个断言陷阱

**别名思路**：fast_1d 与 im2col 的几何、op_params 完全一致，只是 op 标签不同。因此 GPU 后端全部用一行 case 别名到既有 IM2COL 实现，只在 CPU 上提供专用 kernel——这是最低成本的落地方式，核心价值在 CPU kernel 的 O(1) 窗口算法本身。

**坑 1：im2col 不读 kernel。** 初版 CPU kernel 从上游带来一个 `GGML_ASSERT(src0->type == GGML_TYPE_F16)`——在上游项目的上下文里 kernel 恰好总是 F16。但 ggml 原生 `ggml_conv_1d` 走 `dst_type = a->type==BF16 ? F32 : F16`，**kernel（src0）保持 F32**；而 im2col 语义上只展开输入（src1），根本不碰 kernel。删掉该断言后 F32 kernel + F16 dst 立即通过。

**坑 2：conv_1d 的 kernel 布局是 [K, IC, OC]。** 按 PyTorch 习惯建 [K, OC, IC] 会撞上 builder 断言 `b->ne[1] == a->ne[1]`。ggml 的卷积权重处处是「输入通道在 ne[1]」，与 PyTorch [OC, IC, K] 是转置关系。

## 2. conv_transpose_1d_ext：修上游 bug + 布局选择

上游（TTS.cpp）三个问题的修复方式见 [operators_zh.md](operators_zh.md) §2。此处保留移植决策记录：

**布局决策**：groups 用 PyTorch 布局 `a=[K, Cout_pg, Cin_g]`，与生态权重零转换兼容；代价是 `src0->ne[1]` 不再等于输出通道总数——**所有用 `ne01` 当 Cout 的地方都要重新审视**（Vulkan dispatch 的 elements 就是这么翻车的，见 §4）。

**API 兼容**：旧 6 参 `ggml_conv_transpose_1d` 保留，内部委托 ext 版（op0=0, g0=1）。op_params 统一为 4 个 int32。

## 3. SCATTER_ELEMENTS / REL_POS_BIAS：builder 断言的教训

初版 scatter 断言曾允许 `indices->ne[d] == 1`（想模仿 ONNX 的广播），但 CPU/Vulkan kernel 都把 updates 与 indices **平铺成同一线性序列**一起索引——形状不一致时会错位读写。正确约束是逐维 `indices->ne == updates->ne`，且 `updates->ne == data->ne`（除 axis 维）。**断言必须与 kernel 的实际索引方式一致，不能只对齐算子规范。**

**Vulkan 侧的两个专门动作**：

- scatter 的 dst 不是 shader 写出的，而是先 `buffer_copy(data → dst)` 再跑 shader。两次操作之间**必须插 pipelineBarrier**（`TransferWrite → ShaderRead|ShaderWrite`）。copy 用 `ggml_vk_buffer_copy_async` 提交在与 compute 相同的 command buffer 上，屏障保证顺序。
- 累加模式需要 `atomicAdd`：探测 `VK_EXT_shader_atomic_float` 扩展 + `shaderBufferFloat32AtomicAdd` 特性（Turing 及更新的 NVIDIA、多数 RDNA 支持）。探测结果存进 `vk_device`，`supports_op` 据此拒绝；特性链挂载模仿现有 float8 块的 `pNext` 写法，并用 `#if defined(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)` 防旧 SDK 编译失败。

## 4. Vulkan 移植清单与三大坑

`ggml-vulkan.cpp` 是近 2 万行的单体文件，一次移植要按固定清单过一遍：

```
vk_device 成员 / pipeline 成员 / push-constants 结构
→ 扩展探测（在 get_device 内，不是 print_gpu_info！）
→ 特性链 pNext 挂载 + device_extensions.push_back
→ create_pipelines 注册（wg_denoms 要与 shader local_size 一致）
→ op_get_pipeline（按 op + 类型选 pipeline）
→ op_f32 的 elements switch（决定 dispatch 的三维工作组数）
→ op_f32 的 dispatch 分支（决定绑定哪些 buffer）
→ graph compute switch（op → dispatch 函数）
→ supports_op（类型/连续性/参数守卫）
```

**坑 1：`MemoryBarrier` 是 Windows 宏。** `vk::MemoryBarrier(a, b)` 会被 winnt.h 的 `#define MemoryBarrier ...` 撕碎，报出莫名其妙的 `vk::__faststorefence` 不存在。Vulkan-Hpp 代码一律用**花括号聚合初始化**（`{ { src_access, dst_access } }`）绕开——遇到 vulkan.hpp 报怪错先想到 Windows 宏。

**坑 2：elements 语义。** `op_f32` 的 elements 不是"元素数"而是"工作组数 × wg_denoms"的混合约定：`dispatch_pipeline` 做 `ceil(elements[i], wg_denoms[i])`。不同 op 的 case 有各自约定（有的预除 512 配合 shader 内部循环）。**移植时要抄约定，不要发明**——rel_pos_bias 起草时曾用 flat-x 方案（`{ne,1,1}` + `local_size 256`），后换回三维 dispatch（`{W, H, B*HW}` + `8×8×4`），对大输入的 z 轴覆盖更稳。

**坑 3：`src0->ne[1]` ≠ 输出通道数。** convT 的 dispatch elements 原版写 `src0->ne[1]`（无分组时 == Cout）。分组后 `src0->ne[1] = Cout/g0`，于是工作组数不足——**输出张量的后半组全是 0**，且前半组完全正确（正是「错误从中间某元素开始」这一精确现场暴露了它）。修正为 `dst->ne[1]`。教训：**别名/复用既有 op 的 dispatch 时，凡把"src 形状"当"dst 形状"用的地方都要重新审一遍。**

**其他**：

- `ggml_vk_op_f32` 的 push-constant 参数是 `PC &&`，具名变量要 `std::move`。
- `ggml_vk_tensor_subbuffer` 有第三参 `allow_misalign`，copy 类操作传 `true`（散写场景 dst 可能未对齐）。
- 新 `.comp` 文件要同时进 `vulkan-shaders-gen.cpp` 注册（`string_to_spv`）；CMake 用 `file(GLOB ... CONFIGURE_DEPENDS)` 自动拾取文件本身做依赖。
- VS2019 生成器不支持 shader 编译的 `DEPFILE`，Windows 上 Vulkan 构建必须用 Ninja 生成器（见 [building_zh.md](building_zh.md)）。

## 5. 测试策略的收益

`tests/test_learned_ops.c` 按「手写参考值」而非「两实现互相对照」编写（仅 fast_1d 用与原版 conv_1d 同图对照）。这直接抓出了两类问题：

- **布局错误**（[K, OC, IC] vs [K, IC, OC]）在 CPU 第一轮就暴露；
- **分组索引错误**在 Vulkan 上以「前半组对、后半组全 0」的精确模式暴露，直接指向 dispatch 规模而非 shader 数学（shader 数学与 CPU 同构，CPU 通过即说明数学正确——反推唯一能造成"只算了一部分输出"的就是 dispatch 范围）。

**参考值来自三个独立渠道**：convT 的 golden 复核取自上游项目自带测试的数值；scatter/rel_pos 参考直接按算子语义手写；fast_1d 用同图对照。多来源交叉比单来源更不容易"参考本身就错"。

## 6. 测试挂具备忘

- `ggml_cgraph` 对外不透明：用 `ggml_graph_n_nodes` 与 `ggml_graph_node` 遍历；或沿张量 `src[]` 递归。
- v0.19 中 `ggml_backend_alloc_ctx_tensors_from_buft` 不是公开 API；测试用 galloc（中间结果）+ 无 alloc ctx 建图即可。
- 多后端 `ggml_backend_sched` 要求后端列表最后一个必须是 CPU；但在该基线上用 {GPU, CPU} 的 sched 跑混合 op 图出现分配异常（mul_mat 输出与 im2col 输出共享 buffer、节点疑似未执行）——测试改用 **galloc + 单后端 graph_compute** 绕开（这也是 llama.cpp 单 GPU 场景的主流路径）。
- Windows 下程序如可能中途崩溃，`main()` 开头加 `setvbuf(stdout, NULL, _IONBF, 0)`，否则崩溃时缓冲内的全部输出会丢失。

---

# 补丁二移植笔记（qvac 算子）

从单一供体树（tetherto/qvac-ext-ggml 的 `speech` 分支）向更老基线（v0.19.0）批量移植十个算子，踩出了补丁一单算子经验之外的新坑：

## 7. 供体与基线的版本漂移

- **供体树比目标基线新**。qvac 的 ggml 是 v0.20 时代的 dlopen-variant 树（760 个文件不同）。绝不能整文件拷贝；每个算子按"插入点 diff"移植。枚举数值区间不同（供体 `GGML_OP_COUNT` = 107，基线补丁一之前是 104）——`ggml.c` 里**两处** `static_assert(GGML_OP_COUNT == N)` 都要更新，别只改一处。
- **移植前先查基线已有什么**。v0.19 基线的 Vulkan 后端本来就有图级 snake 融合（`snake_pattern` = mul→sin→sqr→mul→add，外加 `snake_f32` pipeline 与 `snake.comp`）。照搬供体导致重复的 `vk_op_snake_push_constants`、重复的 `pipeline_snake_f32` 成员与第二个 `string_to_spv("snake_f32", ...)`——重定义错误加 pipeline 名冲突。正解：**复用上游 pipeline**（其数学与 binding 布局 `{x, a, inv_b, dst}` 与供体完全一致），删除全部重复注册。规则：移植算子 X 之前，先 `git grep -i x` 基线，找出会撞名的名字（shader 名、push-constant 结构体、pipeline 成员、`string_to_spv` 键）。
- **COL2IM_1D 与 ROLL 在 v0.19 上游已存在**——供体的这两个 builder/kernel 不能移植。只有基线枚举里没有的算子才移植。

## 8. Vulkan 重注册清单（每个算子）

新算子必须全部触及，否则编译失败或分发出错：push-constant 结构体 → pipeline 成员 → `ggml_vk_create_pipeline`（对齐 `parameter_count`、`wg_denoms`）→ `vulkan-shaders-gen.cpp` 的 `string_to_spv` → `ggml_vk_op_get_pipeline` case → `ggml_vk_op_f32` 的 `elements` case → 分发 wrapper → graph-compute switch case → `supports_op` →（仅调试）`vk_check_results` clone 链。补丁一的 `elements` 约定备注在这里泛化：通用 unary 组按扁平 512×512 切分计算；按 2-D 索引（`i0 + i1*ne0`）的 shader 需要自己的 `elements = {ne0, ne1, 1}` case，别加进那个组。

## 9. 列主序测试陷阱

- 供体的 CPU 内核按 **ggml 列主序**索引缓冲（`[L, C]` 张量取 `x[t + c*L]`）。用行主序（`x[t*C + c]`）写的参考能编译、能运行、然后处处失败。症状：每个 case 都"错成另一个元素的值"而不是差个量级。改测试的索引，别改内核。
- 多结果测试（同一算子的 layout-0 vs layout-1 变体）必须把**每个结果根**都展开进前向图——`ggml_build_forward_expand` 只拉取你传入的根可达的节点。建了但没作为根（或不可从根到达）的张量在图分配器里**没有 buffer**，第一次对它 `tensor_set` 就死在 `GGML_ASSERT(buf != NULL && "tensor buffer not set")`。三个变体（`_1d`、`_ct`、`_causal_ct`）时三个根都要展开，即使非 causal case 里 causal 那个是 `NULL`。

## 10. PowerShell 生成 patch 的坑

在 PowerShell 5.1 里 `git diff > file` 会写出 **UTF-16 LE**（BOM `FF FE` 让 `git apply` 报 "No valid patches in input"）。用 `cmd /c "git diff ... > file"` 或 `git diff --output=file` 生成，并确认文件头几字节是 ASCII `d i f f`。

## 11. ggml_pad 是按维度加，不是按两侧加

给融合 depthwise 算子写对照组合图时：`ggml_pad(x, p0, p0, 0, 0)` 是给 **ne0 和 ne1 各加 p0**（按维度尺寸），不是给 ne0 两侧各加 p0。same-padding 1D 卷积链直接用 `ggml_conv_1d_dw(w, x, s, p0, d)`——builder 本身收 padding——不要显式 pad。（显式对称 pad 应写 `ggml_pad(x, 2*p0, 0, 0, 0)`。）
