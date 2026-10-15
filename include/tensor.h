#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstdint>

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

#endif // TENSOR_H
