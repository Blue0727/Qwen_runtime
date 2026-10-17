# 架构与数据布局

当前实现针对 Qwen2.5-0.5B Base：单序列、CPU、greedy。本文描述已实现的行为。

## 模型与模块

| 参数 | 数值 |
|---|---:|
| vocab_size | 151936 |
| hidden_size / intermediate_size | 896 / 4864 |
| Transformer 层数 | 24 |
| Query / KV 头数 | 14 / 2 |
| head_dim / kv_dim | 64 / 128 |
| 每个 KV 头对应的 Query 头数 | 7 |
| RoPE theta / RMSNorm epsilon | 1000000 / 0.000001 |

`config_loader` 映射 JSON 配置并验证当前支持范围；`safetensors_loader` 解析头部、张量形状、类型与偏移，读取 BF16 并转换为 FP32；`model` 组织权重、会话状态与生成；`operators` 提供数值算子。

当前加载器读取单个 `model.safetensors`，只接受已实现的 `qwen2` 路径与共享输出权重，不是任意 Hugging Face 模型的通用加载器。

## 权重布局

Linear 权重按 `[out_features, in_features]` 行优先保存，计算为 `y[o] = sum_i(W[o,i] * x[i]) + bias[o]`。

| 权重 | 形状 |
|---|---|
| Embedding / 共享输出投影 | `[151936, 896]`，同一份 FP32 数据 |
| 每层 Q / O 投影 | `[896, 896]` |
| 每层 K / V 投影 | `[128, 896]` |
| FFN gate / up 投影 | `[4864, 896]` |
| FFN down 投影 | `[896, 4864]` |

Q / K / V 带 FP32 bias。Q4 模式只转换上述七个层内投影；Norm、bias 和词表矩阵保持 FP32。`LinearWeight` 使用 `std::variant<TensorF32, TensorQ4>`，统一接口分派相应内核。

加载 Q4 时逐个矩阵读取 FP32 临时数据、量化、释放临时数据，不同时保留全部层的 FP32 权重；每次启动重新量化，不保存量化权重文件。

## 单 token 前向

1. 用 token ID 查 Embedding 得到隐藏向量。
2. 每层先做 RMSNorm，再做 Q / K / V 投影和 bias。
3. 对 Q / K 施加 RoPE，写入当前层、当前位置的 K / V。
4. 执行因果 GQA、O 投影和残差连接。
5. 执行第二次 RMSNorm、SwiGLU FFN 与残差连接。
6. 需要 logits 时，做最终 RMSNorm 和共享词表投影。
7. 全部层完成后，缓存有效长度增加一次。

RMSNorm 为 `y = weight * x / sqrt(mean(x*x) + eps)`；SwiGLU 为 `silu(W_gate*x) * (W_up*x)`，随后用 `W_down` 投影。

每个 Query 头使用 `floor(q_head / 7)` 对应的 KV 头。Attention 分数除以 `sqrt(64)`，再用减去最大值的 softmax 归一化。RoPE 在每个 64 维头内配对 `(i, i+32)`，不是相邻元素配对。

## KV Cache 与 Workspace

K / V 各使用连续 FP32 数组，逻辑形状为 `[layer, position, 128]`，偏移为 `(layer * capacity + position) * 128`。当前两个入口分配 512-token 容量，K + V 数据载荷共 12 MiB；这不是进程内存实测值。

Workspace 预分配隐藏状态、Q / K / V、Attention 输出、FFN 中间向量、分数和 logits，在推理中复用。每次生成重置 KV 有效长度，不重新分配整个缓存，也不复用其他请求的前缀。

Prefill 依次处理输入 token，仅最后一个位置计算最终 Norm / 词表输出；主干仍为逐 token 计算，没有批量 GEMM。第一个新增 token 来自最后一次 Prefill 的 logits；之后每次 Decode forward 预测下一个 token。

达到生成上限时，最后一个输出 token 不再执行 forward。因此无提前 EOS 时，生成 16 个 token 对应 15 次 Decode forward。`cache.length` 统计完成 forward 的 token，不等于页面显示 token 的总数。

## 自定义 Q4

每 32 个连续权重一块，包含 1 个 FP32 scale 和 16 字节打包值，共 20 字节，折合每权重 5 bit（含 scale）。仅比较权重载荷时，相对 FP32 为 6.4 倍压缩，不是整个模型内存缩小 6.4 倍。

非零块量化为 `s = max(abs(w)) / 7`、`q = clamp(round(w/s), -7, 7)`。使用 `std::round`，中间值向远离零方向取整；编码 `u=q+8`。第 j 个字节为 `u[j] | (u[j+16] << 4)`，j 为 0..15。

全零块使用 `scale=0`、打包值 `0x88`。矩阵按行分块，当前要求输入列数是 32 的倍数。

Q4 Linear 直接拆包、减去偏移并计算块点积，乘 scale 后累加，不为整张权重分配反量化 FP32 副本。显式 SIMD 仅用于 FP32，Q4 保留普通 C++ 实现。

## 并行与服务边界

OpenMP 按 Linear 输出行静态分配任务，各线程读取相同输入、计算不同输出行。FP32 点积可使用八通道 SIMD；开关启用时，MSVC 为 `operators.cpp` 设置 `/arch:AVX2`。尚无运行时指令集检测和分派。

常驻 C++ worker 通过 stdin / stdout JSON Lines 接收 token IDs 并返回新增 IDs / 计时。Python 负责分词、整段解码、HTTP 和进程回收，用锁覆盖一次完整收发；网页记录不参与后续推理。当前为单进程、串行请求、本机演示。
