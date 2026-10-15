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
/// @brief 当前的人工检查入口：配置、文件头及一个小型 BF16 张量。
// int main()
// {
//     try
//     {
//         const std::filesystem::path project_root = L"D:/lkm/我的资料/Qwen_runtime";
//         const auto model_dir = project_root / "models" / "Qwen2.5-0.5B";
//         const auto input_path = project_root / "data" / "prompt_ids.json";
//         const auto output_path = project_root / "data" / "generated_ids.json";
//         // 读取分词器脚本输出的整数数组
//         std::ifstream input(input_path);
//         if (!input.is_open())
//         {throw std::runtime_error("Cannot open prompt_ids.json");}
//         const std::vector<int> prompt_tokens = nlohmann::json::parse(input).get<std::vector<int>>();
//         input.close();
//         std::cout << "Loading model...\n" << std::flush;
//         Model model = load_model(model_dir);
//         KVCache cache;
//         cache.init(model.config, 512);
//         Workspace workspace;
//         workspace.init(model.config, cache);
//         constexpr int max_new_tokens = 16;
//         constexpr int eos_token_id = 151643;
//         std::cout << "Prompt tokens: " << prompt_tokens.size() << '\n';
//         std::cout << "Generating...\n" << std::flush;
//         GenerationStats stats;

//         // 预热一次，结果不计入测量。
//         std::cout << "Warmup...\n" << std::flush;
//         generate_greedy(
//             model, prompt_tokens, max_new_tokens,
//             eos_token_id, cache, workspace, stats);

//         std::vector<int> generated;

//         std::cout << std::fixed << std::setprecision(3);
//         for (int run = 1; run <= 3; ++run)
//         {
//             generated = generate_greedy(
//                 model, prompt_tokens, max_new_tokens,
//                 eos_token_id, cache, workspace, stats);

//             std::cout << "\nRun " << run << '\n';
//             std::cout << "Prefill: " << stats.prefill_ms << " ms\n";
//             std::cout << "Decode forwards: "
//                     << stats.decode_forward_calls << '\n';

//             if (stats.decode_forward_calls > 0 && stats.decode_ms > 0.0)
//             {
//                 std::cout << "Decode: "
//                         << stats.decode_ms / stats.decode_forward_calls
//                         << " ms/token, "
//                         << 1000.0 * stats.decode_forward_calls / stats.decode_ms
//                         << " tok/s\n";
//             }
//         }
//         std::ofstream output(output_path);
//         if (!output.is_open())
//         {throw std::runtime_error("Cannot open generated_ids.json");}
//         output << nlohmann::json(generated).dump() << '\n';
//         output.close();
//         if (!output)
//         {
//             throw std::runtime_error("Failed to write generated_ids.json");
//         }

//         std::cout << "Generated tokens: " << generated.size() << '\n';
//         std::cout << "Saved data/generated_ids.json\n";
//         return 0;
//     }
//     catch (const std::exception& e)
//     {
//         std::cerr << "ERROR: " << e.what() << '\n';
//         return 1;    
//     }   

// }

/// @brief 读取 token IDs，执行生成计时并保存续写结果。
int main(int argc, char* argv[])
{
    try
    {
        // 命令行第一个参数指定请求线程数，未指定时使用 4。
        const int thread_count = argc > 1 ? std::stoi(argv[1]) : 4;
        if (thread_count <= 0)
        {
            throw std::invalid_argument("Thread count must be positive");
        }

        // 关闭动态线程数调整，并设置后续并行区域的请求线程数。
        omp_set_dynamic(0);
        omp_set_num_threads(thread_count);

        std::cout << "OpenMP requested threads: "
                  << thread_count << '\n';

        const std::filesystem::path project_root =
            L"D:/lkm/我的资料/Qwen_runtime";

        const auto model_dir =
            project_root / "models" / "Qwen2.5-0.5B";
        const auto input_path =
            project_root / "data" / "prompt_ids.json";
        const auto output_path =
            project_root / "data" / "generated_ids.json";

        std::ifstream input(input_path);
        if (!input.is_open())
        {
            throw std::runtime_error("Cannot open prompt_ids.json");
        }

        const std::vector<int> prompt_tokens =
            nlohmann::json::parse(input).get<std::vector<int>>();
        input.close();

        std::cout << "Loading model...\n" << std::flush;
        Model model = load_model(model_dir);

        KVCache cache;
        cache.init(model.config, 512);

        Workspace workspace;
        workspace.init(model.config, cache);

        constexpr int max_new_tokens = 16;
        constexpr int eos_token_id = 151643;

        std::cout << "Prompt tokens: " << prompt_tokens.size() << '\n';

        GenerationStats stats;

        // 预热一次，不纳入后续三次测量。
        std::cout << "Warmup...\n" << std::flush;
        generate_greedy(
            model, prompt_tokens, max_new_tokens,
            eos_token_id, cache, workspace, stats);

        std::vector<int> generated;
        std::cout << std::fixed << std::setprecision(3);

        // generate_greedy() 每次重置缓存和计时数据。
        for (int run = 1; run <= 3; ++run)
        {
            generated = generate_greedy(
                model, prompt_tokens, max_new_tokens,
                eos_token_id, cache, workspace, stats);

            std::cout << "\nRun " << run << '\n';
            std::cout << "Prefill: " << stats.prefill_ms << " ms\n";
            std::cout << "Decode forwards: "
                      << stats.decode_forward_calls << '\n';

            if (stats.decode_forward_calls > 0 && stats.decode_ms > 0.0)
            {
                std::cout << "Decode: "
                          << stats.decode_ms / stats.decode_forward_calls
                          << " ms/token, "
                          << 1000.0 * stats.decode_forward_calls
                                 / stats.decode_ms
                          << " tok/s\n";
            }
        }

        // 保留最后一次生成结果，供原来的分词器脚本解码。
        std::ofstream output(output_path);
        if (!output.is_open())
        {
            throw std::runtime_error("Cannot open generated_ids.json");
        }

        output << nlohmann::json(generated).dump() << '\n';
        output.close();

        if (!output)
        {
            throw std::runtime_error("Failed to write generated_ids.json");
        }

        std::cout << "Generated tokens: " << generated.size() << '\n';
        std::cout << "Saved data/generated_ids.json\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}