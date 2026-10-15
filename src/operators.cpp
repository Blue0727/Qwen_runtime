#include "operators.h"
#include <cmath>
#include <stdexcept>
#include <cassert>

/**
 * @brief 单 token 的 FP32 线性投影。
 * @param weight 行优先矩阵 [out_features,in_features]。
 * @param input 长度为 in_features 的输入。
 * @param output 已分配为 out_features 的独立输出缓冲区。
 * @param bias 长度为 out_features 的可选偏置，nullptr 表示无偏置。
 * @pre 权重形状、元素数及 bias 已在模型加载时验证；输出不与输入或权重共用存储。
 *
 * 计算公式：y[o] = sum_i(W[o,i] * x[i]) + bias[o]。
 * 文件中的矩阵按输出行存储，每行连续的 in_features 个元素参与一个点积。
 * 缓冲区尺寸由 Workspace 初始化，开发时仅检查调用方是否接错输入或输出。
 */
void linear(
    const TensorF32& weight,
    const std::vector<float>& input,
    std::vector<float>& output,
    const TensorF32* bias)
{
    int out_features = static_cast<int>(weight.shape[0]);
    int in_features = static_cast<int>(weight.shape[1]);
    assert(input.size() == static_cast<std::size_t>(in_features) &&
           output.size() == static_cast<std::size_t>(out_features));

    // W[row,col] 的展开索引为 row * in_features + col。
    // 每行重新从零累加，对应 y[row] = sum_col(W[row,col] * x[col]) + bias[row]。
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < out_features; row++)
    {
        float sum = 0.0f;
        for (int col = 0; col < in_features; col++)
        {
            float w = weight.data[row * in_features + col];
            float x = input[col];
            sum += w * x;
        }

        if (bias != nullptr)
        {
            sum += bias->data[row];
        }
        output[row] = sum;
    }
}

/**
 * @brief 按均方根缩放单个隐藏向量，再乘学习得到的逐维权重。
 * @param weight 已加载、形状为 [H] 的缩放权重。
 * @param input 长度为 H 的输入。
 * @param output 调用方预分配的长度 H 的输出。
 * @param eps 来自已验证配置的有限正数稳定项。
 * @pre 权重及固定配置已验证，H > 0，输出不覆盖模型权重。
 *
 * mean_square = sum_j(input[j]^2) / H
 * output[i] = input[i] * weight[i] / sqrt(mean_square + eps)
 *
 * 两遍计算先得到所有维度共用的缩放系数，再写出各维度结果。
 * 运行时使用 norm_x 接收输出，保留原始 x 供残差连接使用。
 */
void rmsnorm(
    const TensorF32& weight,
    const std::vector<float>& input,
    std::vector<float>& output,
    float eps)
{
    assert(!input.empty() &&
           weight.data.size() == input.size() &&
           output.size() == input.size());

    // 第一遍计算均方值；RMSNorm 不减去输入均值。
    float sum_squares = 0.0f;
    for (size_t i = 0; i < input.size(); i++)
    {
        sum_squares += input[i] * input[i];
    }
    float mean_square = sum_squares / static_cast<float>(input.size());
    // 所有维度共用同一个缩放系数，平方根只计算一次；eps 位于平方根内。
    float inv_rms = 1.0f / std::sqrt(mean_square + eps);
    for (size_t i = 0; i < input.size(); i++)
    {
        output[i] = input[i] * inv_rms * weight.data[i];
    }
}

/**
 * @brief Embedding 查表，复制一个 token 对应的隐藏向量。
 * @param weight 已验证的行优先词表矩阵 [vocab_size,hidden_size]。
 * @param token_id 从 0 开始的词表索引，属于运行时输入。
 * @param output 调用方预分配的长度 hidden_size 的输出。
 * @pre 权重由模型加载器验证，输出缓冲区已初始化。
 * @throws std::invalid_argument token_id 不在词表范围内。
 *
 * output[i] = weight[token_id * hidden_size + i]。
 * token_id 会随输入变化，因此在此入口保留运行时边界检查。
 */
void embedding_lookup(
    const TensorF32& weight,
    int token_id,
    std::vector<float>& output)
{
    std::int64_t vocab_size = weight.shape[0];
    std::int64_t hidden_size = weight.shape[1];
    if (token_id < 0 || token_id >= vocab_size)
    {
        throw std::invalid_argument(
            "Invalid token ID"
        );
    }
    assert(output.size() == static_cast<std::size_t>(hidden_size));

    // 行优先存储中，第 token_id 行从 token_id * hidden_size 开始。
    // 仅复制该行，权重矩阵继续由 Model 持有。
    std::size_t start = static_cast<std::size_t>(token_id) * static_cast<std::size_t>(hidden_size);
    for (std::size_t i = 0; i < static_cast<std::size_t>(hidden_size); i++)
    {
        output[i] = weight.data[start + i];
    }
}

/**
 * @brief 对单 token 的 Q 或 K 原地应用 Qwen RoPE。
 * @param x 连续存储的 [num_heads,head_dim] 向量。
 * @param num_heads Q 或 K 的头数，当前分别为 14、2。
 * @param head_dim 头内维度 D，当前为 64。
 * @param position 从 0 开始的全局 token 位置。
 * @param theta 已验证配置中的频率基数。
 * @pre 配置及位置合法，向量长度为 num_heads * head_dim。
 *
 * 每个 head 配对 (i,i+D/2)，angle = position * theta^(-2*i/D)。
 * a' = a*cos(angle) - b*sin(angle)
 * b' = a*sin(angle) + b*cos(angle)
 *
 * 只旋转 Q/K；历史 K 已旋转后存入缓存，读取时直接使用。
 */
void rope_inplace(
    std::vector<float>& x,
    int num_heads,
    int head_dim,
    int position,
    float theta)
{
    assert(x.size() == static_cast<size_t>(num_heads * head_dim));
    int half = head_dim / 2;
    // 指数中的除法使用浮点计算，先按维度对计算 sin/cos，再供所有 head 共用。
    for (int i = 0; i < half; i++)
    {
        float freq = std::pow(theta, -2.0f * i / head_dim);
        float angle = position * freq;
        float cos_angle = std::cos(angle);
        float sin_angle = std::sin(angle);
        for (int h = 0; h < num_heads; h++)
        {
            // base 定位当前 head；两个索引位于该 head 的前后两半。
            int base = h * head_dim;
            int idx1 = base + i;
            int idx2 = base + half + i;
            // 先保存两个原值，两次写回都依赖它们。
            float a = x[idx1];
            float b = x[idx2];
            x[idx1] = a * cos_angle - b * sin_angle;
            x[idx2] = a * sin_angle + b * cos_angle;
        }
    }
}

/**
 * @brief 原地计算有效前缀的稳定 Softmax。
 * @param values 前 count 项由分数覆盖为概率，其余项保持原值。
 * @param count 有效元素数量，Attention 中为 position + 1。
 * @pre 0 < count <= values.size()，有效前缀的输入分数为有限数。
 *
 * m = max_j(values[j])
 * p[i] = exp(values[i]-m) / sum_j(exp(values[j]-m))
 *
 * 求最大值、指数和分母、归一化各用一遍循环，均限定在 [0,count)。
 * 减去共同最大值保持概率不变，并使指数不大于 1，避免正指数溢出。
 */
void softmax_inplace(
    std::vector<float>& values,
    int count)
{
    assert(count > 0 && static_cast<size_t>(count) <= values.size());
    // 第 1 遍从首项开始取最大值，正确覆盖全部分数为负的情况。
    float max_val = values[0];
    for (int i = 1; i < count; i++)
    {
        if (values[i] > max_val)
        {
            max_val = values[i];
        }
    }
    // 第 2 遍写入指数并累加分母；有限输入的最大项为 exp(0)=1。
    float sum = 0.0f;
    for (int i = 0; i < count; i++)
    {
        values[i] = std::exp(values[i] - max_val);
        sum += values[i];
    }
    // 第 3 遍归一化；缓冲区尾部不参与当前 head 的计算。
    for (int i = 0; i < count; i++)
    {
        values[i] /= sum;
    }
}

// SwiGLU activation
void swiglu_inplace(
    std::vector<float>& gate,
    const std::vector<float>& up
)
{
    assert(gate.size() == up.size());
    for(size_t i = 0; i < gate.size(); i++)
    {
        float x = gate[i];
        float sigmoid;
        //计算sigmoid,避免数值过大溢出
        if(x >= 0)
        {
            float e = std::exp(-x);
            sigmoid = 1.0f / (1.0f + e);
        }
        else
        {
            float e = std::exp(x);
            sigmoid = e / (1.0f + e);
        }
        gate[i] = x * sigmoid * up[i];
    }
}