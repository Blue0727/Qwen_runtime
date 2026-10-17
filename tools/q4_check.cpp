#include "operators.h"
#include "safetensors_loader.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace
{

// 验证程序使用异常报告失败，Release 下同样生效。
void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void print_error(const char* name, const ErrorStats& stats)
{
    std::cout << name
              << ": max_abs=" << stats.max_abs_error
              << ", rmse=" << stats.rmse
              << ", relative_l2=" << stats.relative_l2_error
              << '\n';
}

/**
 * @brief 用已知矩阵验证编码、行偏移和有无 bias 的点积。
 *
 * [2,64] 共四个块，第一个为零块，其余使用不同 scale。
 * 非零块的位置 0 / 16 保存 -7 / 7，因此第一个字节应为 0xF1。
 * 位置 1 / 17 保存 -1 / 1，字节应为 0x97。
 * 位置 15 / 31 保存 3 / -2，字节应为 0x6B。
 *
 * 输入、权重及偏置都能精确表示，这组点积可以精确比较。
 */
void check_known_matrix()
{
    TensorF32 weight;
    weight.shape = {2, 64};
    weight.data.resize(128, 0.0f);

    for (int b = 0; b < 4; ++b)
    {
        const float scale =
            b == 0 ? 0.0f : static_cast<float>(1 << b);
        const int start = b * Q4_BLOCK_SIZE;

        weight.data[start] = -7.0f * scale;
        weight.data[start + 1] = -scale;
        weight.data[start + 15] = 3.0f * scale;
        weight.data[start + 16] = 7.0f * scale;
        weight.data[start + 17] = scale;
        weight.data[start + 31] = -2.0f * scale;
    }

    const TensorQ4 quantized = quantize_q4_matrix(weight);
    const TensorF32 restored = dequantize_q4_matrix(quantized);

    require(quantized.blocks.size() == 4, "Wrong block count");
    require(restored.shape == weight.shape, "Wrong restored shape");
    require(restored.data == weight.data, "Wrong matrix reconstruction");

    for (int b = 0; b < 4; ++b)
    {
        const float expected_scale =
            b == 0 ? 0.0f : static_cast<float>(1 << b);

        std::array<std::uint8_t, Q4_BLOCK_SIZE / 2> expected;
        expected.fill(0x88);

        if (b != 0)
        {
            expected[0] = 0xF1;
            expected[1] = 0x97;
            expected[15] = 0x6B;
        }

        require(
            quantized.blocks[b].scale == expected_scale,
            "Wrong block scale");
        require(
            quantized.blocks[b].packed == expected,
            "Wrong packed layout");
    }

    std::vector<float> input(64);
    for (int i = 0; i < 64; ++i)
    {
        input[i] = static_cast<float>(i % 11 - 5);
    }

    TensorF32 bias;
    bias.shape = {2};
    bias.data = {0.5f, -1.5f};

    for (int mode = 0; mode < 2; ++mode)
    {
        const TensorF32* bias_ptr = mode == 0 ? nullptr : &bias;
        std::vector<float> reference(2);
        std::vector<float> actual(2);

        linear(weight, input, reference, bias_ptr);
        linear_q4(quantized, input, actual, bias_ptr);

        require(reference == actual, "Known matrix Linear mismatch");
    }

    std::cout << "[PASS] Known matrix and packed layout\n";
}

/**
 * @brief 对真实模型矩阵分别统计量化误差和内核计算误差。
 *
 * Q4 内核的参考输出必须来自反量化权重，
 * 这样权重量化误差不会混入内核正确性判断。
 */
void check_model_weight(
    const std::filesystem::path& model_path,
    const SafeTensorsHeader& header,
    const std::string& weight_name,
    const char* bias_name = nullptr)
{
    const TensorF32 weight =
        load_bf16_tensor(model_path, header, weight_name);
    const TensorQ4 quantized = quantize_q4_matrix(weight);
    const TensorF32 restored = dequantize_q4_matrix(quantized);

    TensorF32 bias;
    const TensorF32* bias_ptr = nullptr;

    if (bias_name != nullptr)
    {
        bias = load_bf16_tensor(model_path, header, bias_name);
        bias_ptr = &bias;
    }

    const std::size_t rows =
        static_cast<std::size_t>(weight.shape[0]);
    const std::size_t cols =
        static_cast<std::size_t>(weight.shape[1]);

    // 确定性的非均匀输入，避免全一输入掩盖位置对应错误。
    std::vector<float> input(cols);
    for (std::size_t i = 0; i < cols; ++i)
    {
        input[i] =
            0.25f * std::sin(static_cast<float>(i) * 0.17f);
    }

    std::vector<float> original_output(rows);
    std::vector<float> reference_output(rows);
    std::vector<float> actual_output(rows);

    linear(weight, input, original_output, bias_ptr);
    linear(restored, input, reference_output, bias_ptr);
    linear_q4(quantized, input, actual_output, bias_ptr);

    std::cout << "\n" << weight_name << '\n';

    print_error(
        "Weight quantization",
        compute_error_stats(weight.data, restored.data));
    print_error(
        "Original FP32 vs Q4",
        compute_error_stats(original_output, actual_output));
    print_error(
        "Dequantized FP32 vs Q4",
        compute_error_stats(reference_output, actual_output));

    // 容许两条累加路径的浮点舍入差异；不用于限制量化误差。
    for (std::size_t i = 0; i < rows; ++i)
    {
        const double reference =
            static_cast<double>(reference_output[i]);
        const double actual =
            static_cast<double>(actual_output[i]);

        const double tolerance =
            1e-5 + 1e-4 * std::fabs(reference);

        require(
            std::fabs(actual - reference) <= tolerance,
            "Q4 Linear differs from dequantized FP32 reference");
    }

    std::cout << "[PASS] Q4 Linear reference comparison\n";
}

} // namespace

int main()
{
    try
    {
        omp_set_dynamic(0);
        omp_set_num_threads(4);

        std::cout << std::scientific << std::setprecision(6);

        check_known_matrix();

        // 从项目根目录运行，使用 filesystem 原生路径处理中文目录。
        const auto model_path =
            std::filesystem::current_path() /
            "models" / "Qwen2.5-0.5B" / "model.safetensors";

        const auto header = read_safetensors_header(model_path);

        // 输入长度 896，包含 FP32 bias。
        check_model_weight(
            model_path,
            header,
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias");

        // 输入长度 4864，不带 bias。
        check_model_weight(
            model_path,
            header,
            "model.layers.0.mlp.down_proj.weight");

        std::cout << "\nAll Q4 checks passed.\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}