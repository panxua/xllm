/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>

#include "xllm/core/util/tensor_helper.h"

namespace xllm {
namespace layer {
namespace moe_dump {

inline thread_local int64_t g_current_step = 0;

struct DumpConfig {
  std::string root_dir;
  std::unordered_set<int64_t> steps;
  std::unordered_set<int64_t> layers;
};

inline std::string sanitize_name(std::string name) {
  for (auto& ch : name) {
    if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' ||
          ch == '-' || ch == '.')) {
      ch = '_';
    }
  }
  return name;
}

inline std::unordered_set<int64_t> parse_csv_int_set(const char* env_value,
                                                     int64_t default_value) {
  std::unordered_set<int64_t> values;
  if (env_value == nullptr || env_value[0] == '\0') {
    values.insert(default_value);
    return values;
  }
  std::stringstream ss(env_value);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    try {
      values.insert(std::stoll(token));
    } catch (const std::exception&) {
      LOG(WARNING) << "[MOE][Dump] invalid list token: " << token
                   << ", raw=" << env_value;
    }
  }
  if (values.empty()) {
    values.insert(default_value);
  }
  return values;
}

inline const DumpConfig& get_config() {
  static std::once_flag init_flag;
  static DumpConfig config;
  std::call_once(init_flag, [&]() {
    const char* root = std::getenv("MOE_DUMP_DIR");
    if (root != nullptr && root[0] != '\0') {
      config.root_dir = root;
    }
    config.steps =
        parse_csv_int_set(std::getenv("MOE_DUMP_STEPS"), /*default_value=*/0);
    config.layers =
        parse_csv_int_set(std::getenv("MOE_DUMP_LAYERS"), /*default_value=*/0);
    LOG(INFO) << "[MOE][Dump] dir=" << config.root_dir
              << " steps_size=" << config.steps.size()
              << " layers_size=" << config.layers.size();
  });
  return config;
}

inline bool is_enabled() { return !get_config().root_dir.empty(); }

inline bool should_dump_step(int64_t step) {
  if (!is_enabled()) {
    return false;
  }
  return get_config().steps.count(step) > 0;
}

inline bool should_dump_layer(int32_t layer_id) {
  if (!is_enabled()) {
    return false;
  }
  return get_config().layers.count(layer_id) > 0;
}

inline bool should_dump(int64_t step, int32_t layer_id) {
  return should_dump_step(step) && should_dump_layer(layer_id);
}

inline std::string make_dir(int64_t step, int64_t rank, int32_t layer_id) {
  const auto& cfg = get_config();
  if (cfg.root_dir.empty()) {
    return "";
  }
  const auto dir = cfg.root_dir + "/step" + std::to_string(step) + "/rank" +
                   std::to_string(rank) + "/layer" + std::to_string(layer_id);
  try {
    std::filesystem::create_directories(dir);
  } catch (const std::filesystem::filesystem_error& e) {
    LOG(WARNING) << "[MOE][Dump] failed to create " << dir << ": " << e.what();
    return "";
  }
  return dir;
}

inline void dump_tensor(int64_t step,
                        int64_t rank,
                        int32_t layer_id,
                        const std::string& name,
                        const torch::Tensor& tensor) {
  if (!tensor.defined() || !should_dump(step, layer_id)) {
    return;
  }
  const auto dir = make_dir(step, rank, layer_id);
  if (dir.empty()) {
    return;
  }
  const auto path = dir + "/" + sanitize_name(name) + ".pt";
  try {
    save_tensor_as_pickle(tensor.contiguous().to(torch::kCPU), path);
  } catch (const c10::Error& e) {
    LOG(WARNING) << "[MOE][Dump] failed to save " << path << ": "
                 << e.what_without_backtrace();
  }
}

inline int64_t next_step() {
  static std::atomic<int64_t> counter(0);
  return counter.fetch_add(1, std::memory_order_relaxed);
}

inline void set_current_step(int64_t step) { g_current_step = step; }

inline int64_t current_step() { return g_current_step; }

}  // namespace moe_dump
}  // namespace layer
}  // namespace xllm
