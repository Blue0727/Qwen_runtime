#include "safetensors_loader.h"
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <cstring>

/// @brief 将 BF16 位模式扩展为 IEEE 754 binary32，不进行整数到浮点的数值转换。
float bf16_to_fp32(std::uint16_t bf16_value)
{
    std::uint32_t f32_value = static_cast<std::uint32_t>(bf16_value) << 16;
    float result;
    // memcpy 保留位模式，并避免通过指针强转违反严格别名规则。
    std::memcpy(&result, &f32_value, sizeof(result));
    return result;
}

TensorF32 load_bf16_tensor(
    const std::filesystem::path& model_path,
    const SafeTensorsHeader& header,
    const std::string& tensor_name)
{
    const TensorInfo& info = header.tensors.at(tensor_name);
    if (info.dtype != "BF16")
    {
        throw std::runtime_error("Tensor " + tensor_name + " is not of type bf16");
    }

    TensorF32 tensor;
    tensor.shape = info.shape;
    std::size_t num_elements = 1;
    for (auto dim : info.shape)
    {
        if (dim <= 0)
        {
            throw std::runtime_error("Invalid tensor shape for " + tensor_name);
        }
        num_elements *= static_cast<std::size_t>(dim);
    }

    std::uint64_t tensor_bytes = info.data_end - info.data_begin;
    if (tensor_bytes != num_elements * sizeof(std::uint16_t))
    {
        throw std::runtime_error("Tensor size mismatch for " + tensor_name);
    }

    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open safetensors file: " + model_path.string());
    }

    // 文件头中的偏移相对于数据区，seekg 使用文件绝对偏移。
    std::uint64_t absolute_offset = header.data_start + info.data_begin;
    file.seekg(absolute_offset, std::ios::beg);
    if (!file)
    {
        throw std::runtime_error("Failed to seek to tensor data for " + tensor_name);
    }

    // 只保留当前张量的 BF16 临时数据，避免加载整个权重文件副本。
    // 当前目标为小端 Windows x64，可直接读取 BF16 的 16 位位模式。
    std::vector<std::uint16_t> raw(num_elements);
    file.read(reinterpret_cast<char*>(raw.data()), tensor_bytes);
    if (!file)
    {
        throw std::runtime_error("Failed to read tensor data for " + tensor_name);
    }

    tensor.data.resize(num_elements);
    for (std::size_t i = 0; i < num_elements; ++i)
    {
        tensor.data[i] = bf16_to_fp32(raw[i]);
    }
    return tensor;
}

SafeTensorsHeader read_safetensors_header(const std::filesystem::path& model_path)
{
    SafeTensorsHeader result{};
    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open safetensors file: " + model_path.string());
    }

    result.file_size = std::filesystem::file_size(model_path);
    if (result.file_size < 8)
    {
        throw std::runtime_error("Invalid safetensors file: too small");
    }

    // 前 8 字节是小端 uint64 文件头长度；当前目标为小端 Windows x64。
    std::uint64_t header_length;
    file.read(reinterpret_cast<char*>(&header_length), sizeof(header_length));
    if (!file)
    {
        throw std::runtime_error("Failed to read header length");
    }

    result.header_length = header_length;
    if (header_length > result.file_size - 8)
    {
        throw std::runtime_error("Invalid header length");
    }
    result.data_start = 8 + header_length;

    std::string header_json;
    header_json.resize(header_length);
    file.read(header_json.data(), header_length);
    if (!file)
    {
        throw std::runtime_error("Failed to read JSON header");
    }
    nlohmann::json j = nlohmann::json::parse(header_json);

    for (auto& [name, value] : j.items())
    {
        // __metadata__ 是文件级信息，不包含张量的 dtype、shape 或 data_offsets。
        if (name == "__metadata__")
        {
            continue;
        }

        TensorInfo info{};
        info.dtype = value.at("dtype").get<std::string>();
        info.shape = value.at("shape").get<std::vector<std::int64_t>>();
        auto offsets = value.at("data_offsets").get<std::vector<std::uint64_t>>();
        if (offsets.size() != 2)
        {
            throw std::runtime_error("Invalid data_offsets for tensor: " + name);
        }

        info.data_begin = offsets[0];
        info.data_end = offsets[1];
        if (info.data_begin > info.data_end)
        {
            throw std::runtime_error("Invalid tensor offset: " + name);
        }
        if (info.data_end > result.file_size - result.data_start)
        {
            throw std::runtime_error("Tensor exceeds file size: " + name);
        }
        result.tensors.emplace(name, info);
    }
    return result;
}
