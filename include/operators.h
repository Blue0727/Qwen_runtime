#ifndef OPERATORS_H
#define OPERATORS_H

#include "tensor.h"

#include <vector>

// FP32 线性投影；输出由调用方预分配，可使用一维偏置。
void linear(
    const TensorF32& weight,
    const std::vector<float>& input,
    std::vector<float>& output,
    const TensorF32* bias = nullptr);

// 对单个隐藏向量做 RMSNorm，输出由调用方预分配。
void rmsnorm(
    const TensorF32& weight,
    const std::vector<float>& input,
    std::vector<float>& output,
    float eps);

// 按 token_id 复制 embedding 的一行到已有输出缓冲区。
void embedding_lookup(
    const TensorF32& weight,
    int token_id,
    std::vector<float>& output);

// 对连续存储的 Q 或 K 原地应用 RoPE。
void rope_inplace(
    std::vector<float>& x,
    int num_heads,
    int head_dim,
    int position,
    float theta);

// 原地归一化前 count 个分数，保留缓冲区剩余部分。
void softmax_inplace(
    std::vector<float>& values,
    int count);

// SwiGLU activation
void swiglu_inplace(
    std::vector<float>& gate,
    const std::vector<float>& up
);
#endif // OPERATORS_H
