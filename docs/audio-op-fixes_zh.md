# 音频算子正确性修复（补丁七）

> **[English](audio-op-fixes.md)** | 中文

在现有集合之后应用 `patches/audio-op-fixes-ggml0190.patch`。补丁从已提交的
ggml 工作树差异生成，一至六保持不变。修复对应 pc-nsf-hifigan.cpp PR #2
指出的底层算子问题。

- CPU 和 Vulkan ScatterElements 将 `[-dimension, -1]` 内的负索引归一化。
  越界更新在计算目标地址前丢弃，未更新的位置保留原始 data。覆盖和累加模式
  采用相同规则，Vulkan 对 `INT32_MIN` 也不会发生有符号整数溢出。
- 转置卷积允许与 stride 不满足整除关系的非负 padding，例如 `pad=2,
  stride=3`。各后端原有的能力门控仍然生效。
- CPU 直接卷积按 `w->nb[]` 字节偏移打包权重，按 `bias->nb[0]` 读取 bias，
  正确处理非连续视图。GPU 的连续性要求不变。

`tests/test_audio_op_regressions.cpp` 包含 12 项 CPU 用例：四个 scatter 轴和
两种 reduction、分组/非分组转置卷积的非整除 padding，以及连续/非连续的直接
卷积权重和 bias。测试检查结果有限且符合明确的期望值。集合原有参考函数和
误差阈值均未修改。

使用 `tests/` 中的 CMake 构建后执行：

```bash
ctest --test-dir build-metal-tests --output-on-failure
# 测试构建启用 GGML_VULKAN=ON 后：
build-vulkan-tests/test_audio_op_regressions Vulkan0
```

CPU 验证环境：Apple M4、macOS 27.0、AppleClang 21、Release、2026-09-05：
`CPU: 12 passed, 0 unsupported`。Vulkan 运行验证由 consumer PR 的 Linux/Mesa
CI 任务执行，未在本 macOS 主机上运行。
