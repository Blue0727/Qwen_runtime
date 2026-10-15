#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <cstddef>

/**
 * @brief Qwen2 模型的结构参数，来自 config.json。
 *
 * 修改配置后应先调用 validate()，再使用派生尺寸或分配运行时缓冲区。
 */
struct ModelConfig
{
    int vocab_size;              ///< 词表大小。
    int hidden_size;             ///< 隐藏向量维度。
    int num_layers;              ///< Transformer 层数。
    int num_heads;               ///< Query 注意力头数。
    int num_kv_heads;            ///< Key / Value 注意力头数。
    int intermediate_size;       ///< FFN 中间维度。
    int max_position_embeddings; ///< 模型允许的最大位置范围。
    float rope_theta;            ///< RoPE 频率计算的基数。
    float rms_norm_eps;          ///< RMSNorm 的数值稳定项。
    bool tie_word_embeddings;    ///< 输出投影是否复用 embedding 权重。

    /// @brief 每个注意力头的维度；调用前配置必须通过 validate()。
    int head_dim() const;

    /// @brief 所有 KV 头拼接后的维度；调用前配置必须通过 validate()。
    int kv_dim() const;

    /// @brief GQA 中每个 KV 头对应的 Query 头数；调用前配置必须通过 validate()。
    int q_heads_per_kv_head() const;

    /**
     * @brief 校验尺寸、头数关系、RoPE 参数及当前模型支持范围。
     * @throws std::invalid_argument 配置不满足运行时要求。
     */
    void validate() const;
};

#endif // MODEL_CONFIG_H
