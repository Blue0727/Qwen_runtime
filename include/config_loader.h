#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include "model_config.h"
#include <filesystem>

/**
 * @brief 从 config.json 加载并校验 Qwen2 模型配置。
 * @param config_path 配置文件路径；Windows 中文路径应使用宽字符构造。
 * @return 已通过 ModelConfig::validate() 的配置。
 * @throws std::runtime_error 文件无法打开。
 * @throws std::invalid_argument 模型类型或结构参数不受支持。
 * @throws nlohmann::json::exception JSON 解析、字段访问或类型转换失败。
 */
ModelConfig load_model_config(const std::filesystem::path& config_path);

#endif // CONFIG_LOADER_H
