# Q4 验证

## 方法

入口为 `tools/q4_check.cpp`，使用项目的量化、反量化、误差统计和 Linear。2026-10-17 的 Release 运行通过以下检查；程序请求 4 个 OpenMP 线程并关闭动态调整，该次输出未单独记录 AVX2 开关。

- 已知 `[2,64]` 矩阵：零块、前后半块配对、不同 scale、跨行块位置、反量化及有 / 无 bias 点积。
- 第 0 层 `self_attn.q_proj.weight` / bias，输入维度 896。
- 第 0 层 `mlp.down_proj.weight`，输入维度 4864。

真实权重测试输入为 `x[i] = 0.25 * sin(float(i) * 0.17)`，不是模型执行产生的激活。

三类误差分别统计，以区分量化损失和内核实现误差：

1. 原始权重与反量化权重。
2. 原始 FP32 投影输出与 Q4 输出。
3. **同一组已量化权重**的反量化 FP32 Linear 与直接 Q4 Linear。

第 3 类逐元素门槛为 `abs(actual-reference) <= 1e-5 + 1e-4*abs(reference)`。统计最大绝对误差、RMSE 和相对 L2，误差累加使用 double。该门槛只针对本次 FP32 累加验证，不是文本质量标准。

## 原始输出

```text
[PASS] Known matrix and packed layout

model.layers.0.self_attn.q_proj.weight
Weight quantization: max_abs=8.705357e-02, rmse=7.282066e-03, relative_l2=1.075269e-01
Original FP32 vs Q4: max_abs=2.491587e-01, rmse=3.580449e-02, relative_l2=4.587881e-03
Dequantized FP32 vs Q4: max_abs=1.907349e-06, rmse=1.344724e-07, relative_l2=1.723328e-08
[PASS] Q4 Linear reference comparison

model.layers.0.mlp.down_proj.weight
Weight quantization: max_abs=2.202497e-02, rmse=1.924236e-03, relative_l2=1.039275e-01
Original FP32 vs Q4: max_abs=7.195182e-02, rmse=2.294217e-02, relative_l2=1.041951e-01
Dequantized FP32 vs Q4: max_abs=3.576279e-07, rmse=1.075404e-07, relative_l2=4.865366e-07
[PASS] Q4 Linear reference comparison

All Q4 checks passed.
```

## 结论

所测矩阵和输入上，直接 Q4 内核与反量化 FP32 参考最大绝对误差为 `1.907349e-6`、`3.576279e-7`，符合上述容差。两组权重的相对 L2 量化误差约为 10.75% 和 10.39%，不能解释为文本质量下降相同比例。

这组检查覆盖已测打包、拆包和点积路径，不覆盖所有矩阵、输入或模型接入行为。完整生成已跑通，但没有独立参考框架的全模型数值对齐；质量限制和速度见[基准记录](benchmarks.md)。
