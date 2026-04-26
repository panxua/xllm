/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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
#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sstream>
#include <string>

#include "util/tensor_helper.h"

namespace xllm {
namespace tensor_dump {

struct DumpContext {
  int64_t step = -1;
  int32_t rank = -1;
  int64_t layer = -1;
};

inline thread_local DumpContext g_context;

inline void set_step(int64_t step) { g_context.step = step; }

inline void clear_step() { g_context.step = -1; }

class ScopedStep {
 public:
  explicit ScopedStep(int64_t step) : previous_step_(g_context.step) {
    set_step(step);
  }

  ~ScopedStep() { set_step(previous_step_); }

 private:
  int64_t previous_step_ = -1;
};

class ScopedRankLayer {
 public:
  ScopedRankLayer(int32_t rank, int64_t layer)
      : previous_rank_(g_context.rank), previous_layer_(g_context.layer) {
    g_context.rank = rank;
    g_context.layer = layer;
  }

  ~ScopedRankLayer() {
    g_context.rank = previous_rank_;
    g_context.layer = previous_layer_;
  }

 private:
  int32_t previous_rank_ = -1;
  int64_t previous_layer_ = -1;
};

inline bool env_flag_enabled(const char* value) {
  if (value == nullptr) {
    return false;
  }
  std::string normalized(value);
  std::transform(normalized.begin(),
                 normalized.end(),
                 normalized.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return normalized == "1" || normalized == "true" || normalized == "on" ||
         normalized == "yes";
}

inline bool enabled() {
  return env_flag_enabled(std::getenv("XLLM_DUMP_TENSOR"));
}

inline std::optional<std::filesystem::path> dump_root() {
  const char* root = std::getenv("DUMP_DIR");
  if (root == nullptr || root[0] == '\0') {
    static bool warned = false;
    if (enabled() && !warned) {
      LOG(WARNING) << "XLLM_DUMP_TENSOR is enabled but DUMP_DIR is not set; "
                      "tensor dump is disabled.";
      warned = true;
    }
    return std::nullopt;
  }
  return std::filesystem::path(root);
}

inline bool should_dump(int64_t layer) {
  return enabled() && g_context.step == 0 && layer == 0 &&
         dump_root().has_value();
}

inline bool should_dump_current() {
  return g_context.rank >= 0 && should_dump(g_context.layer);
}

inline std::string tensor_info(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  std::ostringstream os;
  os << "shape=" << tensor.sizes()
     << ", dtype=" << c10::toString(tensor.scalar_type())
     << ", device=" << tensor.device();
  return os.str();
}

inline std::string sanitize_path_component(std::string value) {
  for (char& c : value) {
    const auto uc = static_cast<unsigned char>(c);
    if (!(std::isalnum(uc) || c == '_' || c == '-' || c == '.')) {
      c = '_';
    }
  }
  return value;
}

inline std::filesystem::path tensor_path(int64_t step,
                                         int32_t rank,
                                         int64_t layer,
                                         const std::string& module,
                                         const std::string& name) {
  auto root = dump_root().value();
  return root / ("step" + std::to_string(step)) /
         ("rank" + std::to_string(rank)) / ("layer" + std::to_string(layer)) /
         sanitize_path_component(module) /
         (sanitize_path_component(name) + ".pt");
}

inline std::filesystem::path tensor_path(const std::filesystem::path& root,
                                         int64_t step,
                                         int32_t rank,
                                         int64_t layer,
                                         const std::string& module,
                                         const std::string& name) {
  return root / ("step" + std::to_string(step)) /
         ("rank" + std::to_string(rank)) / ("layer" + std::to_string(layer)) /
         sanitize_path_component(module) /
         (sanitize_path_component(name) + ".pt");
}

inline std::optional<std::filesystem::path> dump_root_or_log_skip(
    int32_t rank,
    int64_t layer,
    const std::string& module,
    const std::string& name,
    const torch::Tensor& tensor) {
  if (!enabled()) {
    VLOG(1) << "[TENSOR_DUMP] skip " << module << "/" << name
            << ": XLLM_DUMP_TENSOR is disabled, rank=" << rank
            << ", step=" << g_context.step << ", layer=" << layer << ", "
            << tensor_info(tensor);
    return std::nullopt;
  }
  if (g_context.step != 0) {
    VLOG(1) << "[TENSOR_DUMP] skip " << module << "/" << name
            << ": current version only dumps step0, rank=" << rank
            << ", step=" << g_context.step << ", layer=" << layer << ", "
            << tensor_info(tensor);
    return std::nullopt;
  }
  if (layer != 0) {
    VLOG(1) << "[TENSOR_DUMP] skip " << module << "/" << name
            << ": current version only dumps layer0, rank=" << rank
            << ", step=" << g_context.step << ", layer=" << layer << ", "
            << tensor_info(tensor);
    return std::nullopt;
  }
  auto root = dump_root();
  if (!root.has_value()) {
    VLOG(1) << "[TENSOR_DUMP] skip " << module << "/" << name
            << ": DUMP_DIR is not set, rank=" << rank
            << ", step=" << g_context.step << ", layer=" << layer << ", "
            << tensor_info(tensor);
    return std::nullopt;
  }
  return root;
}

inline void save_tensor(int32_t rank,
                        int64_t layer,
                        const std::string& module,
                        const std::string& name,
                        const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    VLOG(1) << "[TENSOR_DUMP] skip " << module << "/" << name
            << ": tensor is undefined, rank=" << rank
            << ", step=" << g_context.step << ", layer=" << layer;
    return;
  }

  auto root = dump_root_or_log_skip(rank, layer, module, name, tensor);
  if (!root.has_value()) {
    return;
  }

  try {
    const auto path =
        tensor_path(root.value(), g_context.step, rank, layer, module, name);
    std::filesystem::create_directories(path.parent_path());
    auto saved = tensor.detach().to(torch::kCPU).contiguous();
    save_tensor_as_pickle(saved, path.string());
    VLOG(1) << "[TENSOR_DUMP] saved " << module << "/" << name << " to "
            << path.string() << ", rank=" << rank << ", step=" << g_context.step
            << ", layer=" << layer << ", " << tensor_info(tensor);
  } catch (const c10::Error& e) {
    LOG(ERROR) << "Failed to dump tensor " << module << "/" << name << ": "
               << e.what_without_backtrace();
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to dump tensor " << module << "/" << name << ": "
               << e.what();
  }
}

inline void save_optional_tensor(int32_t rank,
                                 int64_t layer,
                                 const std::string& module,
                                 const std::string& name,
                                 const std::optional<torch::Tensor>& tensor) {
  if (tensor.has_value()) {
    save_tensor(rank, layer, module, name, tensor.value());
  } else {
    VLOG(1) << "[TENSOR_DUMP] skip " << module << "/" << name
            << ": optional tensor has no value, rank=" << rank
            << ", step=" << g_context.step << ", layer=" << layer;
  }
}

inline void save_tensor(const std::string& module,
                        const std::string& name,
                        const torch::Tensor& tensor) {
  if (g_context.rank < 0 || g_context.layer < 0) {
    VLOG(1) << "[TENSOR_DUMP] skip " << module << "/" << name
            << ": dump rank/layer context is not set, rank=" << g_context.rank
            << ", step=" << g_context.step << ", layer=" << g_context.layer
            << ", " << tensor_info(tensor);
    return;
  }
  save_tensor(g_context.rank, g_context.layer, module, name, tensor);
}

inline void save_optional_tensor(const std::string& module,
                                 const std::string& name,
                                 const std::optional<torch::Tensor>& tensor) {
  if (g_context.rank < 0 || g_context.layer < 0) {
    VLOG(1) << "[TENSOR_DUMP] skip " << module << "/" << name
            << ": dump rank/layer context is not set, rank=" << g_context.rank
            << ", step=" << g_context.step << ", layer=" << g_context.layer;
    return;
  }
  if (!tensor.has_value()) {
    VLOG(1) << "[TENSOR_DUMP] skip " << module << "/" << name
            << ": optional tensor has no value, rank=" << g_context.rank
            << ", step=" << g_context.step << ", layer=" << g_context.layer;
    return;
  }
  save_tensor(module, name, tensor.value());
}

}  // namespace tensor_dump
}  // namespace xllm
