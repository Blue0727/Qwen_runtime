#include "config_loader.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>

ModelConfig load_model_config(const std::filesystem::path& config_path)
{
    std::ifstream file(config_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open config file" + config_path.string());
    }
    nlohmann::json j = nlohmann::json::parse(file);

    std::string model_type = j.at("model_type").get<std::string>();
    if (model_type != "qwen2")
    {
        throw std::invalid_argument("Unsupported model_type: " + model_type);
    }

    // 显式映射 Hugging Face 字段名，保留缺失字段和类型不匹配的异常。
    ModelConfig cfg{};
    cfg.vocab_size = j.at("vocab_size").get<int>();
    cfg.hidden_size = j.at("hidden_size").get<int>();
    cfg.num_layers = j.at("num_hidden_layers").get<int>();
    cfg.num_heads = j.at("num_attention_heads").get<int>();
    cfg.num_kv_heads = j.at("num_key_value_heads").get<int>();
    cfg.intermediate_size = j.at("intermediate_size").get<int>();
    cfg.max_position_embeddings = j.at("max_position_embeddings").get<int>();
    cfg.rope_theta = j.at("rope_theta").get<float>();
    cfg.rms_norm_eps = j.at("rms_norm_eps").get<float>();
    cfg.tie_word_embeddings = j.at("tie_word_embeddings").get<bool>();

    cfg.validate();
    return cfg;
}
