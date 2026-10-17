#include <iostream>
#include <filesystem>
#include <iomanip>
#include "safetensors_loader.h"
#include "config_loader.h"
#include "model.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <omp.h>
#include <string>
/// @brief 读取指定 token IDs，执行生成计时并保存续写结果。
int main(int argc, char* argv[])
{
    try
    {
        const int thread_count =
            argc > 1 ? std::stoi(argv[1]) : 4;
        const std::string weight_mode =
            argc > 2 ? argv[2] : "fp32";
        const int max_new_tokens =
            argc > 3 ? std::stoi(argv[3]) : 16;

        if (thread_count <= 0)
        {
            throw std::invalid_argument(
                "Thread count must be positive");
        }

        if (weight_mode != "fp32" && weight_mode != "q4")
        {
            throw std::invalid_argument(
                "Weight mode must be fp32 or q4");
        }

        if (max_new_tokens <= 0)
        {
            throw std::invalid_argument(
                "Max new tokens must be positive");
        }

        const WeightFormat weight_format =
            weight_mode == "q4"
                ? WeightFormat::Q4
                : WeightFormat::FP32;

        const std::filesystem::path project_root =
            L"D:/lkm/我的资料/Qwen_runtime";

        const auto model_dir =
            project_root / "models" / "Qwen2.5-0.5B";

        // 相对输入路径以项目根目录解释；绝对路径可直接使用。
        const auto input_path =
            argc > 4
                ? project_root / std::filesystem::path(argv[4])
                : project_root / "data" / "prompt_ids.json";

        // 文件名区分输入、权重模式和生成上限，保留不同实验结果。
        std::filesystem::path output_name("generated_");
        output_name += input_path.stem();
        output_name +=
            "_" + weight_mode + "_" +
            std::to_string(max_new_tokens) + ".json";

        const auto output_path =
            project_root / "data" / output_name;

        std::ifstream input(input_path);
        if (!input)
        {
            throw std::runtime_error(
                "Cannot open input: " + input_path.u8string());
        }

        const std::vector<int> prompt_tokens =
            nlohmann::json::parse(input).get<std::vector<int>>();
        input.close();

        omp_set_dynamic(0);
        omp_set_num_threads(thread_count);

        std::cout << "Layer Linear weights: "
                  << weight_mode << '\n';
        std::cout << "OpenMP requested threads: "
                  << thread_count << '\n';

        // 模型只加载一次；Q4 模式在这里完成量化。
        std::cout << "Loading model...\n" << std::flush;
        Model model = load_model(model_dir, weight_format);

        KVCache cache;
        cache.init(model.config, 512);

        Workspace workspace;
        workspace.init(model.config, cache);

        constexpr int eos_token_id = 151643;

        std::cout << "Input: "
                  << input_path.filename().u8string() << '\n';
        std::cout << "Prompt tokens: "
                  << prompt_tokens.size() << '\n';
        std::cout << "Max new tokens: "
                  << max_new_tokens << '\n';

        GenerationStats stats;

        // 预热一次；generate_greedy 每次都会 reset 会话状态。
        std::cout << "Warmup...\n" << std::flush;
        generate_greedy(
            model, prompt_tokens, max_new_tokens,
            eos_token_id, cache, workspace, stats);

        std::vector<int> generated;
        std::cout << std::fixed << std::setprecision(3);

        for (int run = 1; run <= 3; ++run)
        {
            generated = generate_greedy(
                model, prompt_tokens, max_new_tokens,
                eos_token_id, cache, workspace, stats);

            std::cout << "\nRun " << run << '\n';
            std::cout << "Prefill: "
                      << stats.prefill_ms << " ms\n";
            std::cout << "Decode forwards: "
                      << stats.decode_forward_calls << '\n';

            if (stats.decode_forward_calls > 0 &&
                stats.decode_ms > 0.0)
            {
                std::cout
                    << "Decode: "
                    << stats.decode_ms / stats.decode_forward_calls
                    << " ms/token, "
                    << 1000.0 * stats.decode_forward_calls
                           / stats.decode_ms
                    << " tok/s\n";
            }
        }

        // 保留最后一次生成的 IDs，供分词器桥接解码。
        std::ofstream output(output_path);
        if (!output)
        {
            throw std::runtime_error(
                "Cannot open output: " + output_path.u8string());
        }

        output << nlohmann::json(generated).dump() << '\n';
        output.close();

        if (!output)
        {
            throw std::runtime_error(
                "Failed to write generated token IDs");
        }

        std::cout << "Generated tokens: "
                  << generated.size() << '\n';
        std::cout << "Saved data/"
                  << output_path.filename().u8string() << '\n';

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}