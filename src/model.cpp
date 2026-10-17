#include "model.h"
#include "config_loader.h"
#include "operators.h"
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <utility>

/**
 * @brief 按模型要求校验形状，并加载为拥有内存的 FP32 张量。
 * @param expected_shape 根据已验证的配置确定的行优先存储形状。
 * @note 文件读取、BF16 类型和字节长度检查由 load_bf16_tensor() 负责。
 */
static TensorF32 load_checked_tensor(
    const std::filesystem::path& model_path,
    const SafeTensorsHeader& header,
    const std::string& tensor_name,
    const std::vector<std::int64_t>& expected_shape)
{
    // 缺失权重由 at() 报错；形状在读取数据前验证。
    const TensorInfo& info = header.tensors.at(tensor_name);
    if (info.shape != expected_shape)
    {
        throw std::runtime_error("Tensor " + tensor_name + " shape mismatch");
    }
    return load_bf16_tensor(model_path, header, tensor_name);
}

/**
 * @brief 加载一个投影矩阵，并转换为指定的存储表示。
 *
 * 每次只处理一个矩阵：
 * FP32 模式移动数据到 variant，避免复制；
 * Q4 模式返回量化结果，函数退出时释放临时 FP32 数据。
 *
 * 形状、文件类型和数据范围检查复用现有加载与量化入口。
 */
static LinearWeight load_linear_weight(
    const std::filesystem::path& model_path,
    const SafeTensorsHeader& header,
    const std::string& tensor_name,
    const std::vector<std::int64_t>& expected_shape,
    WeightFormat weight_format)
{
    TensorF32 fp32 = load_checked_tensor(
        model_path, header, tensor_name, expected_shape);

    if (weight_format == WeightFormat::Q4)
    {
        return LinearWeight{quantize_q4_matrix(fp32)};
    }

    return LinearWeight{std::move(fp32)};
}

/**
 * @brief 加载一层权重，按指定格式保存七个投影矩阵。
 * @pre config 已验证，layer_index 有效，header 与文件对应。
 *
 * Norm 和 Q/K/V bias 保持 FP32。
 * 投影矩阵逐个加载并转换，避免保留整层 FP32 临时副本。
 */
LayerWeights load_layer_weights(
    const std::filesystem::path& model_path,
    const SafeTensorsHeader& header,
    const ModelConfig& config,
    int layer_index,
    WeightFormat weight_format)
{
    LayerWeights layer;

    const int H = config.hidden_size;
    const int K = config.kv_dim();
    const int I = config.intermediate_size;

    const std::string prefix =
        "model.layers." + std::to_string(layer_index) + ".";

    layer.attn_norm = load_checked_tensor(
        model_path, header,
        prefix + "input_layernorm.weight", {H});

    // Q：[H,H]；K/V：[K,H]，三个偏置保持 FP32。
    layer.Wq = load_linear_weight(
        model_path, header,
        prefix + "self_attn.q_proj.weight", {H, H}, weight_format);
    layer.bq = load_checked_tensor(
        model_path, header,
        prefix + "self_attn.q_proj.bias", {H});

    layer.Wk = load_linear_weight(
        model_path, header,
        prefix + "self_attn.k_proj.weight", {K, H}, weight_format);
    layer.bk = load_checked_tensor(
        model_path, header,
        prefix + "self_attn.k_proj.bias", {K});

    layer.Wv = load_linear_weight(
        model_path, header,
        prefix + "self_attn.v_proj.weight", {K, H}, weight_format);
    layer.bv = load_checked_tensor(
        model_path, header,
        prefix + "self_attn.v_proj.bias", {K});

    layer.Wo = load_linear_weight(
        model_path, header,
        prefix + "self_attn.o_proj.weight", {H, H}, weight_format);

    layer.ffn_norm = load_checked_tensor(
        model_path, header,
        prefix + "post_attention_layernorm.weight", {H});

    // FFN：gate/up 为 [I,H]，down 为 [H,I]。
    layer.W_gate = load_linear_weight(
        model_path, header,
        prefix + "mlp.gate_proj.weight", {I, H}, weight_format);
    layer.W_up = load_linear_weight(
        model_path, header,
        prefix + "mlp.up_proj.weight", {I, H}, weight_format);
    layer.W_down = load_linear_weight(
        model_path, header,
        prefix + "mlp.down_proj.weight", {H, I}, weight_format);

    return layer;
}

/**
 * @brief 加载模型，按指定格式保存层内投影矩阵。
 *
 * Embedding、共享输出投影及所有 Norm / bias 保持 FP32。
 * Q4 转换仅发生在加载阶段，不进入 token forward。
 */
Model load_model(
    const std::filesystem::path& model_dir,
    WeightFormat weight_format)
{
    Model model;

    const auto config_path = model_dir / "config.json";
    const auto model_path = model_dir / "model.safetensors";

    model.config = load_model_config(config_path);
    const SafeTensorsHeader header =
        read_safetensors_header(model_path);

    const int H = model.config.hidden_size;
    const int vocab = model.config.vocab_size;

    // 同一份 FP32 embedding 同时用于最终词表投影。
    model.weights.embed_tokens = load_checked_tensor(
        model_path, header,
        "model.embed_tokens.weight", {vocab, H});

    model.weights.final_norm = load_checked_tensor(
        model_path, header,
        "model.norm.weight", {H});

    model.weights.layers.reserve(model.config.num_layers);

    for (int i = 0; i < model.config.num_layers; ++i)
    {
        model.weights.layers.push_back(
            load_layer_weights(
                model_path,
                header,
                model.config,
                i,
                weight_format));
    }

    return model;
}

/**
 * @brief 分配 [num_layers,capacity,kv_dim] 的 K/V 数组，重置有效长度。
 * @pre config 已验证；容量边界在此初始化入口检查一次。
 */
void KVCache::init(const ModelConfig& config, int content_capacity)
{
    if (content_capacity <= 0 || content_capacity > config.max_position_embeddings)
    {
        throw std::invalid_argument("Invalid KVCache capacity");
    }
    num_layers = config.num_layers;
    kv_dim = config.kv_dim();
    capacity = content_capacity;
    length = 0;
    // 每个数组包含 L * C * K 个 float；乘法在 size_t 中完成。
    std::size_t count = static_cast<std::size_t>(num_layers) *
                        static_cast<std::size_t>(capacity) *
                        static_cast<std::size_t>(kv_dim);
    keys.resize(count);
    values.resize(count);
}

/**
 * @brief 返回整条位置向量的起点，单位是 float 元素索引。
 * @pre 层号和位置在已分配范围内，当前待写位置 length 也可访问。
 */
std::size_t KVCache::offset(int layer_index, int position) const
{
    // 等价于 (layer_index * capacity + position) * kv_dim。
    // 该位置的某个 head 起点为此结果加 kv_head * head_dim。
    return static_cast<std::size_t>(layer_index) *
           static_cast<std::size_t>(capacity) *
           static_cast<std::size_t>(kv_dim) +
           static_cast<std::size_t>(position) *
           static_cast<std::size_t>(kv_dim);
}

/// @brief 重置会话有效长度，复用已有内存。
void KVCache::reset()
{
    // 新请求从位置 0 写入；后续读取仅覆盖新请求的有效前缀。
    length = 0;
}

/**
 * @brief 设置单 token 计算缓冲区大小，后续逐层、逐 token 复用。
 * @pre config 已验证，kv_cache 已使用同一配置初始化。
 */
void Workspace::init(const ModelConfig& config, const KVCache& kv_cache)
{
    int H = config.hidden_size;
    int K = config.kv_dim();
    int I = config.intermediate_size;
    int C = kv_cache.capacity;

    // x 保留残差；Norm 与各投影使用独立输出，避免覆盖尚需读取的输入。
    x.resize(H);
    norm_x.resize(H);
    q.resize(H);
    k.resize(K);
    v.resize(K);
    attn_out.resize(H);
    proj_out.resize(H);
    // FFN 两条支路需要 I 个元素，其输出投影仍复用 H 长度的 proj_out。
    gate.resize(I);
    up.resize(I);
    // scores 供每个 head 轮流使用；计算时仅处理前 position+1 项。
    scores.resize(kv_cache.capacity);
    // 最终共享词表投影写入 V 个 logits，供后续选 token 使用。
    logits.resize(config.vocab_size);
}

/**
 * @brief 单 token 的 GQA：共享 KV head，但分别计算各 Q head 的注意力分布。
 * @param config 已验证的模型配置。
 * @param cache 同一模型的缓存，本层 0..position 的旋转后 K 和原始 V 已写入。
 * @param workspace q 是当前 token 的旋转后 Q；scores 是临时分数 / 概率空间。
 * @param layer_index 当前 Transformer 层号。
 * @param position 当前 token 的位置 p，与历史位置一起参与本层 Attention。
 * @pre 配置、缓存和 Workspace 已初始化；层号、位置和缓冲区尺寸合法。
 *
 * 设 h 为 Q head，g = h / group，D 为 head_dim：
 * s_h(t) = sum_d(Q[h,d] * K[layer,t,g,d]) / sqrt(D)
 * P_h(t) = Softmax(s_h(0),...,s_h(p))[t]
 * O_h(d) = sum_t(P_h(t) * V[layer,t,g,d])，t 的范围为 0..p。
 *
 * 当前 Qwen 为 14 个 Q head、2 个 KV head：group=7。
 * Q head 0..6 共用 KV head 0，7..13 共用 KV head 1，各自保留独立的概率分布。
 * 输出按 Q head 顺序拼接为 [14,64]，即 [hidden_size]。
 *
 * 本函数复用缓冲区并只读缓存。Q/K/V 投影、RoPE、缓存写入由调用流程完成，
 * cache.length 在完整 forward 成功完成所有层后再增加一次。
 */
void gqa_attention(
    const ModelConfig& config,
    const KVCache& cache,
    Workspace& workspace,
    int layer_index,
    int position)
{
    assert(position >= 0 && position < cache.capacity);

    int D = config.head_dim();
    int group = config.q_heads_per_kv_head();
    // 缩放系数在各 head 和位置间共用，当前 D=64 时为 1/8。
    float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // 后面使用 += 累加，必须清除上一层或上一 token 留在输出中的结果。
    std::fill(workspace.attn_out.begin(), workspace.attn_out.end(), 0.0f);
    int num_heads = config.num_heads;
    for (int q_head = 0; q_head < num_heads; q_head++)
    {
        // Q 连续排列为 [q_head,D]；GQA 用整数除法定位共享的 KV head。
        int q_base = q_head * D;
        int kv_head = q_head / group;

        // 第 1 阶段：计算 t=0..position 的缩放点积，包含当前 token。
        // 此处写满 Softmax 将读取的 position+1 项，避免复用上次残留的概率。
        for (int t = 0; t <= position; t++)
        {
            // offset() 定位 [layer,t,kv_dim] 的起点，再跳到内部的 KV head。
            // kv_base 是 float 元素索引，后面的 d 只在该 head 的 D 维内移动。
            std::size_t kv_base = cache.offset(layer_index, t) + kv_head * D;
            float score = 0.0f;
            for (int d = 0; d < D; d++)
            {
                score += workspace.q[q_base + d] * cache.keys[kv_base + d];
            }
            workspace.scores[t] = score * scale;
        }

        // 第 2 阶段：将有效前缀覆盖为概率；各 Q head 轮流复用同一个 scores。
        softmax_inplace(workspace.scores, position + 1);

        // 第 3 阶段：概率乘 V，结果累加到当前 Q head 的独立输出片段。
        // t 包含当前位置；未来位置不参与计算，因此满足单 token 因果约束。
        for (int t = 0; t <= position; t++)
        {
            std::size_t kv_base = cache.offset(layer_index, t) + kv_head * D;
            float weight = workspace.scores[t];
            for (int d = 0; d < D; d++)
            {
                workspace.attn_out[q_base + d] += weight * cache.values[kv_base + d];
            }
        }
    }
}

void attention_forward(
    const ModelConfig& config,
    const LayerWeights& layer,
    KVCache& cache,
    Workspace& workspace,
    int layer_index,
    int position
)
{
    /*
        Attention流程:
        x-RMSNorm-Q/K/V-Linear-RoPE-KV Cache-GQA Attention-Wo
    */
    rmsnorm(layer.attn_norm, workspace.x, workspace.norm_x, config.rms_norm_eps);
    linear(layer.Wq, workspace.norm_x, workspace.q, &layer.bq);
    linear(layer.Wk, workspace.norm_x, workspace.k, &layer.bk);
    linear(layer.Wv, workspace.norm_x, workspace.v, &layer.bv);
    rope_inplace(workspace.q, config.num_heads, config.head_dim(), position, config.rope_theta);
    rope_inplace(workspace.k, config.num_kv_heads, config.head_dim(), position, config.rope_theta);
    std::size_t base = cache.offset(layer_index, position);
    int kv_dim = config.kv_dim();
    for(int i = 0; i < kv_dim; i++)
    {
        cache.keys[base + i] = workspace.k[i];
        cache.values[base + i] = workspace.v[i];
    }
    gqa_attention(config, cache, workspace, layer_index, position);
    linear(layer.Wo, workspace.attn_out, workspace.proj_out);
    
}

// @brief FFN分支forward
// 输入:workspace.x
// 输出:workspace.proj_out，保留x原值
void ffn_forward(const ModelConfig& config,
    const LayerWeights& layer,
    Workspace& workspace)
{
    rmsnorm(layer.ffn_norm, workspace.x, workspace.norm_x, config.rms_norm_eps);
    linear(layer.W_gate, workspace.norm_x, workspace.gate);
    linear(layer.W_up, workspace.norm_x, workspace.up);
    swiglu_inplace(workspace.gate, workspace.up);
    linear(layer.W_down, workspace.gate, workspace.proj_out);
}

// @brief 单层forward，包含Attention和FFN
// 原地更新workspace.x，作为下一层的输入
void forward_layer(
    const ModelConfig& config,
    const LayerWeights& layer,
    KVCache& cache,
    Workspace& workspace,
    int layer_index,
    int position)
{
    attention_forward(config, layer, cache, workspace, layer_index, position);
    for(int i = 0; i < workspace.x.size(); i++)
    {
        workspace.x[i] += workspace.proj_out[i];
    }
    ffn_forward(config, layer, workspace);
    for(int i = 0; i < workspace.x.size(); i++)
    {
        workspace.x[i] += workspace.proj_out[i];
    }
}

/**
 * @brief 完成单个 token 的全模型前向计算。
 * @pre 模型已加载，cache 和 workspace 已使用同一模型配置初始化。
 * @note 输出为原始 logits；成功后 cache.length 增加一次。
 */
void forward_token(
    const Model& model,
    int token_id,
    KVCache& cache,
    Workspace& workspace,
    bool compute_logits)
{
    // 使用引用访问配置和权重，避免复制模型数据。
    const ModelConfig& config = model.config;
    const ModelWeights& weights = model.weights;

    // 在写入缓存前检查容量；token 范围由 embedding_lookup 检查。
    if (cache.length >= cache.capacity)
    {
        throw std::out_of_range("KV cache capacity exceeded");
    }

    // 本次所有层都处理同一个位置，计算期间保持 length 不变。
    const int position = cache.length;

    // E[token_id, :] -> x，得到当前 token 的初始隐藏状态。
    embedding_lookup(weights.embed_tokens, token_id, workspace.x);

    // 各层原地更新 x，并写入各自对应位置的 K/V。
    for (int layer_index = 0;
         layer_index < config.num_layers;
         ++layer_index)
    {
        forward_layer(
            config,
            weights.layers[layer_index],
            cache,
            workspace,
            layer_index,
            position);
    }

    // 所有 Transformer 层之后的最终 RMSNorm：
    // norm_x[h] = final_norm[h] * x[h] / sqrt(mean(x*x) + eps)。
    if(compute_logits)
    {
        rmsnorm(
        weights.final_norm,
        workspace.x,
        workspace.norm_x,
        config.rms_norm_eps);

        // 输出投影与 Embedding 共享 E，形状为 [vocab_size, hidden_size]：
        // logits[v] = sum_h(E[v,h] * norm_x[h])。
        linear(
            weights.embed_tokens,
            workspace.norm_x,
            workspace.logits);
    }
    // 当前 token 完成全部计算后，提交新的缓存有效长度。
    ++cache.length;
}

/**
 * @brief 使用 greedy 策略选择下一个 token。
 * @param logits 完整词表的原始预测分数。
 * @return 最大分数对应的 token ID。
 * @pre logits 非空，元素为有限数，词表大小在 int 范围内。
 *
 * next_token = argmax_v(logits[v])
 *
 * Softmax 保持分数的大小关系，因此可以直接比较原始 logits。
 * 最大值并列时保留第一次出现的位置，使结果确定。
 */
int greedy_select(const std::vector<float>& logits)
{
    assert(!logits.empty());
    std::size_t best_id = 0;
    float best_score = logits[0];
    for(std::size_t i = 1; i < logits.size(); ++i)
    {
        if(logits[i] > best_score)
        {best_score = logits[i]; best_id = i;}
    }
    return static_cast<int>(best_id);
}

/**
 * @brief 对输入 prompt 执行 prefill，并通过 greedy 策略生成后续 token。
 * @param prompt_tokens 已经编码的非空输入序列。
 * @param max_new_tokens 最多生成的 token 数，不包含 prompt。
 * @param eos_token_id 结束符 ID；当前 Qwen2.5-0.5B Base 为 151643。
 * @return 新生成的 token IDs，不包含 prompt 和 EOS。
 * @pre model 已加载，cache 和 workspace 已使用同一配置初始化。
 * 每次调用都重置缓存，表示一次新请求。
 * prompt 与新增 token 的总数不超过 cache.capacity。
 * 达到生成数量上限时，最后一个输出 token 不再执行 forward；
 * cache.length 始终只统计已经完成 forward 的 token。
 */
std::vector<int> generate_greedy(
    const Model& model,
    const std::vector<int>& prompt_tokens,
    int max_new_tokens,
    int eos_token_id,
    KVCache& cache,
    Workspace& workspace,
    GenerationStats& stats)
{
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;
    stats = {};
    if (prompt_tokens.empty())
    {
        throw std::invalid_argument("Prompt must not be empty");
    }
    if (max_new_tokens < 0)
    {
        throw std::invalid_argument("max_new_tokens must not be negative");
    }
    if (prompt_tokens.size() >
        static_cast<std::size_t>(cache.capacity))
    {
        throw std::out_of_range("Prompt exceeds KV cache capacity");
    }
    const int remaining =
        cache.capacity - static_cast<int>(prompt_tokens.size());
    const int limit = std::min(max_new_tokens, remaining);

    // 每次生成都是新请求，复用已分配的缓存内存。
    cache.reset();

    std::vector<int> generated;
    if (limit == 0)
    {
        return generated;
    }
    generated.reserve(static_cast<std::size_t>(limit));
    // Prefill：逐个输入 prompt，建立各层的历史 K/V。
    const auto prefill_begin = Clock::now();
    for(std::size_t i = 0; i < prompt_tokens.size(); ++i)
    {
        const bool compute_logits = (i + 1 == prompt_tokens.size());
        forward_token(model, prompt_tokens[i], cache, workspace, compute_logits);
    }
    stats.prefill_ms = Milliseconds(Clock::now() - prefill_begin).count();
    // 最后一次 forward 的 logits 用于预测第一个新增 token。
    for(int step = 0; step < limit; ++step)
    {
        const int next_token = greedy_select(workspace.logits);
        if(next_token == eos_token_id){break;}
        generated.push_back(next_token);
        if(step + 1 < limit)
        {
            const auto decode_begin = Clock::now();
            forward_token(model, next_token, cache, workspace);
            stats.decode_ms += Milliseconds(Clock::now() - decode_begin).count();
            ++stats.decode_forward_calls;
        }
    }
    return generated;

}


