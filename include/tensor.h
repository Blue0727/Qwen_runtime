#ifndef TENSOR_H
#define TENSOR_H
#include <array>
#include <vector>
#include <cstdint>
#include <variant>

/**
 * @brief 拥有数据内存的连续 FP32 张量，采用行优先布局。
 *
 * shape 描述各维度，data 保存展开后的元素；使用前需保证数据数量与形状一致。
 */
struct TensorF32
{
    std::vector<std::int64_t> shape; ///< 各维度长度。
    std::vector<float> data;         ///< 连续存储的 FP32 元素。
};

inline constexpr int Q4_BLOCK_SIZE = 32;
// 自定义 Q4 块：一个缩放系数和 32 个四位整数码。
struct BlockQ4
{
    float scale = 0.0f;
    std::array<std::uint8_t, Q4_BLOCK_SIZE / 2> packed{};
};

static_assert(sizeof(BlockQ4) == 20, "Unexpected BlockQ4 layout");

// 行优先 Q4 权重矩阵，每行沿输入维度按 32 项分块。
struct TensorQ4
{
    std::vector<std::int64_t> shape; ///< 原始形状：[out_features,in_features]。
    std::vector<BlockQ4> blocks;     ///< 按输出行依次存放的量化块。
};

// 单个投影权重保存 FP32 或 Q4 中的一种表示。
using LinearWeight = std::variant<TensorF32, TensorQ4>;
#endif // TENSOR_H
