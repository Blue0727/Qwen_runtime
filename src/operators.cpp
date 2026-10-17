#include "operators.h"
#include <cmath>
#include <stdexcept>
#include <cassert>
#include <cstddef>
#include <algorithm>
#include <limits>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace
{

/**
 * @brief 计算两个连续 FP32 数组的点积。
 * @pre 两个数组均至少包含 count 个有效元素。
 *
 * 向量路径分八个通道分别累加：
 * acc[k] = a[k] * b[k] + a[k+8] * b[k+8] + ...
 * 最后合并八个通道，并处理不足八项的尾部。
 */
float dot_product_f32(const float* a, const float* b, int count)
{
    float sum = 0.0f;
    int i = 0;

#if defined(__AVX2__)
    // 一个 256 位向量容纳八个 32 位浮点数。
    __m256 acc = _mm256_setzero_ps();

    for (; i + 8 <= count; i += 8)
    {
        // loadu 不要求数组地址按 32 字节对齐。
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);

        const __m256 products = _mm256_mul_ps(va, vb);
        acc = _mm256_add_ps(acc, products);
    }

    // acc 中仍是八个独立的部分和，需要合并为一个标量。
    float lanes[8];
    _mm256_storeu_ps(lanes, acc);

    for (int lane = 0; lane < 8; ++lane)
    {
        sum += lanes[lane];
    }
#endif

    // 向量路径处理尾部；未启用向量路径时处理整个数组。
    for (; i < count; ++i)
    {
        sum += a[i] * b[i];
    }

    return sum;
}

} // namespace

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
    for (int row = 0; row < out_features; ++row)
    {
        // 行优先矩阵中，第 row 行包含连续的 in_features 个元素。
        const float* row_data =
            weight.data.data() +
            static_cast<std::size_t>(row) * in_features;

        // 每个线程独立计算自己负责的输出行。
        float sum = dot_product_f32(
            row_data, input.data(), in_features);

        if (bias != nullptr)
        {
            sum += bias->data[row];
        }

        output[row] = sum;
    }
}

/**
 * @brief 使用 Q4 权重完成单 token 线性投影。
 * @pre weight 为有效的行优先二维矩阵，输入列数为 32 的倍数；
 *      可选 bias ，输入与输出使用独立缓冲区。
 *
 * 每行沿输入方向划分为若干块，每块独立使用 scale：
 * y[row] = sum_b(scale[b] * sum_j(q[b,j] * input[b*32+j]))
 *          + bias[row]
 *
 * 每个字节保存两个量化编码：
 * 低四位对应块内位置 j，高四位对应位置 j+16。
 * 编码减去固定偏移 8，得到有符号量化值。
 *
 * 输出行之间通过 OpenMP 并行；每个线程只写自己的输出元素。
 */
void linear_q4(
    const TensorQ4& weight,
    const std::vector<float>& input,
    std::vector<float>& output,
    const TensorF32* bias)
{
    const int out_features =
        static_cast<int>(weight.shape[0]);
    const int in_features =
        static_cast<int>(weight.shape[1]);

    assert(input.size() == static_cast<std::size_t>(in_features) &&
           output.size() == static_cast<std::size_t>(out_features));

    const std::size_t blocks_per_row =
        static_cast<std::size_t>(in_features) / Q4_BLOCK_SIZE;

    constexpr int half = Q4_BLOCK_SIZE / 2;

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < out_features; ++row)
    {
        // 定位当前输出行的第一个量化块。
        const BlockQ4* row_blocks =
            weight.blocks.data() +
            static_cast<std::size_t>(row) * blocks_per_row;

        float sum = 0.0f;

        for (std::size_t b = 0; b < blocks_per_row; ++b)
        {
            const BlockQ4& block = row_blocks[b];

            // 当前权重块对应输入中连续的 32 个元素。
            const float* x =
                input.data() + b * Q4_BLOCK_SIZE;

            float block_sum = 0.0f;

            for (int j = 0; j < half; ++j)
            {
                const std::uint8_t packed = block.packed[j];

                const int q0 =
                    static_cast<int>(packed & 0x0F) - 8;
                const int q1 =
                    static_cast<int>(packed >> 4) - 8;

                // 两个半块分别与对应位置的输入相乘。
                block_sum += static_cast<float>(q0) * x[j];
                block_sum += static_cast<float>(q1) * x[j + half];
            }

            // 同一块共享 scale，累计后只乘一次。
            sum += block.scale * block_sum;
        }

        if (bias != nullptr)
        {
            sum += bias->data[row];
        }

        output[row] = sum;
    }
}

/**
 * @brief 根据投影权重的存储类型选择对应计算内核。
 * @pre weight 持有有效矩阵，输入输出满足底层算子的要求。
 *
 * 类型分派在每次矩阵调用时执行一次，不进入逐行或逐块循环。
 * 通过 const 指针或引用访问实际权重，不复制或反量化矩阵。
 */
void linear(
    const LinearWeight& weight,
    const std::vector<float>& input,
    std::vector<float>& output,
    const TensorF32* bias)
{
    // 持有 FP32 时返回其地址，否则返回 nullptr。
    if (const TensorF32* fp32 =
            std::get_if<TensorF32>(&weight))
    {
        linear(*fp32, input, output, bias);
        return;
    }

    // 当前另一种权重类型为 Q4，get 返回 const 引用。
    linear_q4(
        std::get<TensorQ4>(weight),
        input,
        output,
        bias);
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

/**
 * @brief 将连续 32 个 FP32 权重量化为一个 Q4 块。
 * @pre input 至少包含 Q4_BLOCK_SIZE 个有效元素。
 *
 * scale = max_i(abs(input[i])) / 7
 * q[i] = clamp(round(input[i] / scale), -7, 7)
 *
 * 将 q[i] 加 8 转为四位无符号编码：
 * packed[j] 的低四位保存第 j 项，高四位保存第 j+16 项。
 */
BlockQ4 quantize_q4_block(const float* input)
{
    constexpr int half = Q4_BLOCK_SIZE / 2;

    float max_abs = 0.0f;

    // 在量化入口检查非有限值，避免将 NaN / Inf 转成整数。
    for (int i = 0; i < Q4_BLOCK_SIZE; ++i)
    {
        const float value = input[i];

        if (!std::isfinite(value))
        {
            throw std::invalid_argument(
                "Q4 quantization requires finite weights");
        }

        max_abs = std::max(max_abs, std::fabs(value));
    }

    BlockQ4 block;
    block.scale = max_abs / 7.0f;

    // q=0 对应编码 8，因此两个零组成字节 0x88。
    block.packed.fill(0x88);

    // 全零块直接返回，同时避免 scale 下溢为零时发生除零。
    if (block.scale == 0.0f)
    {
        return block;
    }

    for (int j = 0; j < half; ++j)
    {
        // 前半块与后半块使用相同的缩放系数，分别量化。
        const int q0 = std::clamp(
            static_cast<int>(
                std::round(input[j] / block.scale)),
            -7, 7);

        const int q1 = std::clamp(
            static_cast<int>(
                std::round(input[j + half] / block.scale)),
            -7, 7);

        const int low = q0 + 8;
        const int high = q1 + 8;

        // 前半块放低四位，后半块放高四位。
        block.packed[j] =
            static_cast<std::uint8_t>(low | (high << 4));
    }

    return block;
}

/**
 * @brief 将一个 Q4 块近似还原为连续 32 个 FP32 权重。
 * @pre block 为有效块，output 至少容纳 Q4_BLOCK_SIZE 项。
 *
 * 拆包得到四位编码，减去固定偏移 8，再乘块缩放系数：
 * output[j]    = scale * (low - 8)
 * output[j+16] = scale * (high - 8)
 */
void dequantize_q4_block(const BlockQ4& block, float* output)
{
    constexpr int half = Q4_BLOCK_SIZE / 2;

    for (int j = 0; j < half; ++j)
    {
        const std::uint8_t packed = block.packed[j];

        const int q0 = static_cast<int>(packed & 0x0F) - 8;
        const int q1 = static_cast<int>(packed >> 4) - 8;

        // 按量化时的前后半块配对，写回原始位置。
        output[j] = block.scale * static_cast<float>(q0);
        output[j + half] = block.scale * static_cast<float>(q1);
    }
}

/**
 * @brief 将二维 FP32 权重矩阵转换为行优先 Q4 矩阵。
 * @pre 源数据数量与 shape 一致，由加载入口保证。
 *
 * 每行划分为 cols/32 个块，各块独立计算 scale。
 * 第 row 行的第 b 块：
 *   FP32 起点 = row * cols + b * Q4_BLOCK_SIZE
 *   Q4 块索引 = row * blocks_per_row + b
 */
TensorQ4 quantize_q4_matrix(const TensorF32& weight)
{
    // 在转换入口检查当前 Q4 格式要求的形状。
    if (weight.shape.size() != 2 ||
        weight.shape[0] <= 0 ||
        weight.shape[1] <= 0 ||
        weight.shape[1] % Q4_BLOCK_SIZE != 0)
    {
        throw std::invalid_argument(
            "Q4 requires a positive matrix with columns divisible by 32");
    }

    const std::size_t rows =
        static_cast<std::size_t>(weight.shape[0]);
    const std::size_t cols =
        static_cast<std::size_t>(weight.shape[1]);
    const std::size_t blocks_per_row = cols / Q4_BLOCK_SIZE;

    assert(weight.data.size() == rows * cols);

    TensorQ4 result;
    result.shape = weight.shape;
    result.blocks.resize(rows * blocks_per_row);

    for (std::size_t row = 0; row < rows; ++row)
    {
        // 分别定位当前行的 FP32 数据和 Q4 块数组。
        const float* source_row =
            weight.data.data() + row * cols;
        BlockQ4* target_row =
            result.blocks.data() + row * blocks_per_row;

        for (std::size_t b = 0; b < blocks_per_row; ++b)
        {
            target_row[b] = quantize_q4_block(
                source_row + b * Q4_BLOCK_SIZE);
        }
    }

    return result;
}

/**
 * @brief 将 Q4 矩阵近似还原为相同形状的 FP32 矩阵。
 * @pre weight 为量化或加载入口保证有效的二维 Q4 矩阵，
 *      维度为正，列数为 Q4_BLOCK_SIZE 的倍数。
 *
 * 每个块还原到当前输出行中对应的连续 32 项。
 * 该函数用于误差比较，后续 Q4 Linear 将直接读取量化块。
 */
TensorF32 dequantize_q4_matrix(const TensorQ4& weight)
{
    const std::size_t rows =
        static_cast<std::size_t>(weight.shape[0]);
    const std::size_t cols =
        static_cast<std::size_t>(weight.shape[1]);
    const std::size_t blocks_per_row = cols / Q4_BLOCK_SIZE;

    assert(weight.blocks.size() == rows * blocks_per_row);

    TensorF32 result;
    result.shape = weight.shape;
    result.data.resize(rows * cols);

    for (std::size_t row = 0; row < rows; ++row)
    {
        const BlockQ4* source_row =
            weight.blocks.data() + row * blocks_per_row;
        float* target_row =
            result.data.data() + row * cols;

        for (std::size_t b = 0; b < blocks_per_row; ++b)
        {
            dequantize_q4_block(
                source_row[b],
                target_row + b * Q4_BLOCK_SIZE);
        }
    }

    return result;
}

/**
 * @brief 比较两组非空、等长的 FP32 数值。
 *
 * e[i] = actual[i] - reference[i]
 * RMSE = sqrt(sum_i(e[i]^2) / N)
 * relative_l2 = sqrt(sum_i(e[i]^2) / sum_i(reference[i]^2))
 *
 * 转为 double 后再相减和累加，减少统计本身的舍入误差。
 * 参考数据全零时：误差也为零则返回 0，否则相对误差为正无穷。
 */
ErrorStats compute_error_stats(
    const std::vector<float>& reference,
    const std::vector<float>& actual)
{
    if (reference.empty() || reference.size() != actual.size())
    {
        throw std::invalid_argument(
            "Error statistics require equally sized nonempty arrays");
    }

    ErrorStats stats;
    double error_square_sum = 0.0;
    double reference_square_sum = 0.0;

    for (std::size_t i = 0; i < reference.size(); ++i)
    {
        const double ref = static_cast<double>(reference[i]);
        const double value = static_cast<double>(actual[i]);
        const double error = value - ref;

        // 验证阶段拒绝非有限结果，避免 NaN 被最大值统计掩盖。
        if (!std::isfinite(error))
        {
            throw std::runtime_error(
                "Cannot measure error with non-finite values");
        }

        stats.max_abs_error =
            std::max(stats.max_abs_error, std::fabs(error));

        error_square_sum += error * error;
        reference_square_sum += ref * ref;
    }

    const double count = static_cast<double>(reference.size());
    stats.rmse = std::sqrt(error_square_sum / count);

    if (reference_square_sum > 0.0)
    {
        stats.relative_l2_error =
            std::sqrt(error_square_sum / reference_square_sum);
    }
    else
    {
        stats.relative_l2_error =
            error_square_sum == 0.0
                ? 0.0
                : std::numeric_limits<double>::infinity();
    }

    return stats;
}