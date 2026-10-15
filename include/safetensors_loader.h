#ifndef SAFETENSORS_LOADER_H
#define SAFETENSORS_LOADER_H

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <unordered_map>
#include "tensor.h"

/// @brief 单个张量在 safetensors 文件中的元数据，不持有权重数据。
struct TensorInfo
{
    std::string dtype;                  ///< 文件中的数据类型，如 "BF16"。
    std::vector<std::int64_t> shape;    ///< 张量形状。
    std::uint64_t data_begin;           ///< 相对数据区起点的起始字节偏移。
    std::uint64_t data_end;             ///< 相对数据区起点的结束偏移，不包含该位置。
};

/// @brief 文件头与张量索引；张量偏移均相对于 data_start。
struct SafeTensorsHeader
{
    std::uint64_t header_length;        ///< JSON 文件头的字节长度。
    std::uint64_t data_start;           ///< 数据区的文件绝对偏移：8 + header_length。
    std::uint64_t file_size;            ///< 整个文件的字节长度。
    std::unordered_map<std::string, TensorInfo> tensors;///< 按权重名称查询元数据。
};

/**
 * @brief 读取 safetensors 文件头，不加载权重数据。
 * @param model_path 模型权重文件路径。
 * @return 文件信息与张量索引，跳过非张量条目 __metadata__。
 * @throws std::runtime_error 文件读取失败或偏移范围非法。
 * @throws std::filesystem::filesystem_error 获取文件大小失败。
 * @throws nlohmann::json::exception JSON 文件头解析或字段访问失败。
 */
SafeTensorsHeader read_safetensors_header(const std::filesystem::path& model_path);

/**
 * @brief 按名称读取一个 BF16 张量，并转换为拥有内存的 FP32 张量。
 * @param model_path 模型权重文件路径。
 * @param header 同一文件的解析结果；解析后文件内容应保持不变。
 * @param tensor_name 文件头中的完整权重名称。
 * @return 保留原始形状与行优先布局的 FP32 张量。
 * @throws std::out_of_range 文件头中不存在指定张量。
 * @throws std::runtime_error 类型、形状、字节长度检查失败，或数据读取失败。
 */
TensorF32 load_bf16_tensor(
    const std::filesystem::path& model_path,
    const SafeTensorsHeader& header,
    const std::string& tensor_name);

#endif // SAFETENSORS_LOADER_H
