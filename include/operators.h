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

// Q4 权重、FP32 输入输出；输出由调用方预分配。
void linear_q4(
    const TensorQ4& weight,
    const std::vector<float>& input,
    std::vector<float>& output,
    const TensorF32* bias = nullptr);

// 根据权重类型选择 FP32 或 Q4 线性投影。
void linear(
    const LinearWeight& weight,
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

// 将连续 32 个 FP32 权重量化为一个 Q4 块。
BlockQ4 quantize_q4_block(const float* input);

// 将一个 Q4 块还原到调用方提供的 32 项缓冲区。
void dequantize_q4_block(const BlockQ4& block, float* output);

// 量化二维权重矩阵，要求列数为 32 的倍数。
TensorQ4 quantize_q4_matrix(const TensorF32& weight);

// 将有效 Q4 矩阵近似还原为 FP32，用于误差检查。
TensorF32 dequantize_q4_matrix(const TensorQ4& weight);

// 两组等长数值的误差统计。
struct ErrorStats
{
    double max_abs_error = 0.0;     ///< 最大绝对误差。
    double rmse = 0.0;              ///< 均方根误差。
    double relative_l2_error = 0.0; ///< 相对 L2 误差。
};

ErrorStats compute_error_stats(
    const std::vector<float>& reference,
    const std::vector<float>& actual);


#endif // OPERATORS_H
