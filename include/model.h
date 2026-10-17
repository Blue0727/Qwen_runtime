#ifndef MODEL_H
#define MODEL_H

#include "operators.h"
#include "tensor.h"
#include "model_config.h"
#include "safetensors_loader.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

// Transformer 层内线性投影的权重格式。
enum class WeightFormat
{
    FP32,
    Q4
};
// 单层权重。H=hidden_size，K=kv_dim，I=intermediate_size。
struct LayerWeights
{
    TensorF32 attn_norm;  ///< Attention RMSNorm：[H]。
    LinearWeight Wq;     ///< Query 投影：[H,H]。
    TensorF32 bq;        ///< Query 偏置：[H]。
    LinearWeight Wk;     ///< Key 投影：[K,H]。
    TensorF32 bk;        ///< Key 偏置：[K]。
    LinearWeight Wv;     ///< Value 投影：[K,H]。
    TensorF32 bv;        ///< Value 偏置：[K]。
    LinearWeight Wo;     ///< Attention 输出投影：[H,H]。
    TensorF32 ffn_norm;  ///< FFN RMSNorm：[H]。
    LinearWeight W_gate; ///< SwiGLU gate：[I,H]。
    LinearWeight W_up;   ///< SwiGLU up：[I,H]。
    LinearWeight W_down; ///< FFN 输出投影：[H,I]。
};

// 全模型权重，推理期间只读使用。
struct ModelWeights
{
    TensorF32 embed_tokens;            ///< Embedding 与输出投影共用：[vocab_size,H]。
    std::vector<LayerWeights> layers;  ///< 按层号排列的权重。
    TensorF32 final_norm;              ///< 最终 RMSNorm 缩放权重：[H]。
};

// 单次会话的 KV 存储和有效长度。
struct KVCache
{
    int num_layers = 0; ///< 层数。
    int capacity = 0;   ///< token 容量。
    int kv_dim = 0;     ///< 所有 KV 头的拼接长度。
    int length = 0;     ///< 已完成全部层计算的 token 数。
    std::vector<float> keys;   ///< 旋转后的 K。
    std::vector<float> values; ///< 原始 V。

    // 分配缓存并重置有效长度。
    void init(const ModelConfig& config, int content_capacity);

    // 重置有效长度，复用已分配内存。
    void reset();

    // 返回某层、某位置的 KV 向量起点，单位为 float 元素。
    std::size_t offset(int layer_index, int position) const;
};

// 单 token forward 复用的工作缓冲区。
struct Workspace
{
    std::vector<float> x;        ///< 隐藏状态及残差：[hidden_size]。
    std::vector<float> norm_x;   ///< RMSNorm 输出：[hidden_size]。
    std::vector<float> q;        ///< Q：[hidden_size]。
    std::vector<float> k;        ///< K：[kv_dim]。
    std::vector<float> v;        ///< V：[kv_dim]。
    std::vector<float> attn_out; ///< Attention 结果：[hidden_size]。
    std::vector<float> proj_out; ///< Attention / FFN 输出投影：[hidden_size]。
    std::vector<float> gate;     ///< SwiGLU gate 支路：[intermediate_size]。
    std::vector<float> up;       ///< SwiGLU up 支路：[intermediate_size]。
    std::vector<float> scores;   ///< 复用的分数 / 概率缓冲区：[capacity]。
    std::vector<float> logits;   ///< 最终词表输出：[vocab_size]。

    // 配置和缓存已初始化时，设置各缓冲区大小。
    void init(const ModelConfig& config, const KVCache& kv_cache);
};

// 模型配置及权重；会话状态单独持有。
struct Model
{
    ModelConfig config;
    ModelWeights weights;
};

// 加载指定层，按指定格式保存七个投影矩阵。
LayerWeights load_layer_weights(
    const std::filesystem::path& model_path,
    const SafeTensorsHeader& header,
    const ModelConfig& config,
    int layer_index,
    WeightFormat weight_format);

// 加载模型，默认保留 FP32 投影权重。
Model load_model(
    const std::filesystem::path& model_dir,
    WeightFormat weight_format = WeightFormat::FP32);

// 使用旋转后的 Q 和已写入的当前层缓存，计算 attn_out。
void gqa_attention(
    const ModelConfig& config,
    const KVCache& cache,
    Workspace& workspace,
    int layer_index,
    int position);

// Attention分支forward
// 输入:workspace.x
// 输出:workspace.proj_out
// x保持不变，用于Residual
void attention_forward(
    const ModelConfig& config,
    const LayerWeights& layer,
    KVCache& cache,
    Workspace& workspace,
    int layer_index,
    int position
);

// FFN分支forward
// 输入:workspace.x  
// 输出:workspace.proj_out，保留x原值
void ffn_forward(
    const ModelConfig& config,
    const LayerWeights& layer,
    Workspace& workspace);

// 单层forward，包含Attention和FFN
// 原地更新workspace.x，作为下一层的输入
void forward_layer(
    const ModelConfig& config,
    const LayerWeights& layer,
    KVCache& cache,
    Workspace& workspace,
    int layer_index,
    int position);

// 处理一个 token，将下一 token 的预测分数写入 workspace.logits。
void forward_token(
    const Model& model,
    int token_id,
    KVCache& cache,
    Workspace& workspace,
    bool compute_logits = true);

struct GenerationStats
{
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    int decode_forward_calls = 0;
};

// 返回最大Logit对应的token_id：并列时选择较小的token_id
int greedy_select(const std::vector<float>& logits);

// 开始一次新的greedy生成，返回新增 token，不包含prompt和EOS
std::vector<int> generate_greedy(
    const Model& model,
    const std::vector<int>& prompt_tokens,
    int max_new_tokens,
    int eos_token_id,
    KVCache& cache,
    Workspace& Workspace,
    GenerationStats& stats);




#endif // MODEL_H
