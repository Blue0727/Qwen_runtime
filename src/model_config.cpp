#include "model_config.h"
#include <stdexcept>
#include <cmath>
#include <sstream>

int ModelConfig::head_dim() const
{
    return hidden_size / num_heads;
}

int ModelConfig::kv_dim() const
{
    return num_kv_heads * head_dim();
}

int ModelConfig::q_heads_per_kv_head() const
{
    return num_heads / num_kv_heads;
}

void ModelConfig::validate() const
{
    // 先校验正尺寸，避免后续取模和派生尺寸计算发生除零。
    if (vocab_size <= 0)
    {
        throw std::invalid_argument("vocab_size must be greater than 0");
    }
    if (hidden_size <= 0)
    {
        throw std::invalid_argument("hidden_size must be greater than 0");
    }
    if (num_layers <= 0)
    {
        throw std::invalid_argument("num_layers must be greater than 0");
    }
    if (num_heads <= 0)
    {
        throw std::invalid_argument("num_heads must be greater than 0");
    }
    if (num_kv_heads <= 0)
    {
        throw std::invalid_argument("num_kv_heads must be greater than 0");
    }
    if (intermediate_size <= 0)
    {
        throw std::invalid_argument("intermediate_size must be greater than 0");
    }
    if (max_position_embeddings <= 0)
    {
        throw std::invalid_argument("max_position_embeddings must be greater than 0");
    }

    if (hidden_size % num_heads != 0)
    {
        std::ostringstream oss;
        oss << "hidden_size (" << hidden_size
            << ") must be divisible by num_heads (" << num_heads << ")";
        throw std::invalid_argument(oss.str());
    }

    // GQA 要求每个 KV 头对应相同数量的 Query 头。
    if (num_heads % num_kv_heads != 0)
    {
        std::ostringstream oss;
        oss << "num_heads (" << num_heads
            << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
        throw std::invalid_argument(oss.str());
    }

    // Qwen 的 RoPE 将头内维度分成两半进行配对旋转。
    if (head_dim() % 2 != 0)
    {
        std::ostringstream oss;
        oss << "head_dim (" << head_dim() << ") must be even for RoPE";
        throw std::invalid_argument(oss.str());
    }

    if (!std::isfinite(rope_theta) || rope_theta <= 0)
    {
        throw std::invalid_argument("rope_theta must be a finite positive number");
    }
    if (!std::isfinite(rms_norm_eps) || rms_norm_eps <= 0)
    {
        throw std::invalid_argument("rms_norm_eps must be a finite positive number");
    }

    // 当前模型的输出投影复用 embedding 权重。
    if (!tie_word_embeddings)
    {
        throw std::invalid_argument("tie_word_embeddings must be true for current model");
    }
}
