#include "model.h"

#include <nlohmann/json.hpp>
#include <omp.h>

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Json = nlohmann::json;

/**
 * @brief 持续处理生成请求，直到标准输入关闭。
 *
 * 模型、缓存和工作区在请求之间复用。
 * generate_greedy() 每次重置缓存有效长度，开始一次独立生成。
 *
 * 标准输出只写 JSON，供后续 Python 服务读取；
 * 每行输出后立即 flush，避免结果留在缓冲区中。
 */
static void run_chat_worker(const Model& model)
{
    KVCache cache;
    cache.init(model.config, 512);

    Workspace workspace;
    workspace.init(model.config, cache);

    // 当前使用 Qwen2.5-0.5B Base 的结束符。
    constexpr int eos_token_id = 151643;

    std::cout << Json{
        {"type", "ready"},
        {"context_capacity", cache.capacity}
    }.dump() << std::endl;

    std::string line;

    while (std::getline(std::cin, line))
    {
        Json response;

        try
        {
            const Json request = Json::parse(line);

            const auto prompt_tokens =
                request.at("input_ids").get<std::vector<int>>();

            const int max_new_tokens =
                request.value("max_new_tokens", 128);

            if (max_new_tokens <= 0)
            {
                throw std::invalid_argument(
                    "max_new_tokens must be positive");
            }

            // 网页请求要求明确预留输出空间，超出预算时返回错误。
            // 空输入、非法 token 等检查继续交给已有生成流程。
            if (prompt_tokens.size() +
                    static_cast<std::size_t>(max_new_tokens) >
                static_cast<std::size_t>(cache.capacity))
            {
                throw std::invalid_argument(
                    "Prompt + max_new_tokens exceeds KV cache capacity");
            }

            GenerationStats stats;

            const auto generated = generate_greedy(
                model,
                prompt_tokens,
                max_new_tokens,
                eos_token_id,
                cache,
                workspace,
                stats);

            response = {
                {"type", "result"},
                {"output_ids", generated},
                {"prefill_ms", stats.prefill_ms},
                {"decode_ms", stats.decode_ms},
                {"decode_forward_calls", stats.decode_forward_calls}
            };
        }
        catch (const std::exception& e)
        {
            // 单条请求失败后仍继续接收下一条请求。
            response = {
                {"type", "error"},
                {"message", e.what()}
            };
        }

        std::cout << response.dump() << std::endl;
    }
}

int main(int argc, char* argv[])
{
    try
    {
        const int thread_count =
            argc > 1 ? std::stoi(argv[1]) : 16;

        const std::string weight_mode =
            argc > 2 ? argv[2] : "q4";

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

        const WeightFormat weight_format =
            weight_mode == "q4"
                ? WeightFormat::Q4
                : WeightFormat::FP32;

        const std::filesystem::path project_root =
            L"D:/lkm/我的资料/Qwen_runtime";

        omp_set_dynamic(0);
        omp_set_num_threads(thread_count);

        // 日志写 stderr，避免混入标准输出的 JSON 通信。
        std::cerr << "Loading model..." << std::endl;

        // 加载放在请求循环之外，整个进程只执行一次。
        const Model model = load_model(
            project_root / "models" / "Qwen2.5-0.5B",
            weight_format);

        run_chat_worker(model);

        return 0;
    }
    catch (const std::exception& e)
    {
        // 启动失败时退出，由后续 Python 服务报告启动错误。
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}