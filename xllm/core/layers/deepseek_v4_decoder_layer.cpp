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

#include "deepseek_v4_decoder_layer.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <sstream>
#include <string>

#include "kernels/ops_api.h"
#include "layers/npu_torch/moe_dump_utils.h"
#include "xllm/core/util/tensor_helper.h"

namespace xllm {
namespace layer {

namespace {

int64_t tensor_bytes(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return 0;
  }
  return tensor.numel() * tensor.element_size();
}

int64_t module_registered_tensor_bytes(const torch::nn::Module& module) {
  int64_t total_bytes = 0;
  for (const auto& item : module.named_parameters(/*recurse=*/true)) {
    total_bytes += tensor_bytes(item.value());
  }
  for (const auto& item : module.named_buffers(/*recurse=*/true)) {
    total_bytes += tensor_bytes(item.value());
  }
  return total_bytes;
}

std::string tensor_shape_string(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  std::ostringstream os;
  os << "[";
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    if (i > 0) {
      os << ",";
    }
    os << tensor.size(i);
  }
  os << "]";
  return os.str();
}

std::string tensor_dtype_device_string(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  std::ostringstream os;
  os << tensor.scalar_type() << "," << tensor.device();
  return os.str();
}

std::string sanitize_dump_name(std::string name) {
  for (auto& ch : name) {
    if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' ||
          ch == '-' || ch == '.')) {
      ch = '_';
    }
  }
  return name;
}

std::string deepseek_v4_dump_root() {
  const char* disabled = std::getenv("XLLM_DSV4_ROPE_DUMP_DISABLE");
  if (disabled != nullptr && std::string(disabled) == "1") {
    return "";
  }
  const char* root = std::getenv("XLLM_DSV4_ROPE_DUMP_DIR");
  if (root != nullptr && root[0] != '\0') {
    return root;
  }
  return "./xllm_deepseek_v4_rope_dump";
}

int32_t deepseek_v4_dump_layer_id() {
  const char* env_layer_id = std::getenv("XLLM_DSV4_DUMP_LAYER_ID");
  if (env_layer_id == nullptr || env_layer_id[0] == '\0') {
    return 0;
  }
  try {
    return std::stoi(env_layer_id);
  } catch (const std::exception&) {
    LOG(WARNING) << "[DSV4][Dump] invalid XLLM_DSV4_DUMP_LAYER_ID="
                 << env_layer_id << ", fallback to 0";
    return 0;
  }
}

std::string make_layer_dump_dir(int64_t tp_rank, int32_t layer_id) {
  const auto root = deepseek_v4_dump_root();
  if (root.empty()) {
    return "";
  }
  const auto dir = root + "/tp_" + std::to_string(tp_rank) + "/layer_" +
                   std::to_string(layer_id);
  try {
    std::filesystem::create_directories(dir);
  } catch (const std::filesystem::filesystem_error& e) {
    LOG(WARNING) << "[DSV4][Dump] failed to create " << dir << ": " << e.what();
    return "";
  }
  return dir;
}

torch::Tensor dump_tensor_on_cpu(const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return torch::Tensor();
  }
  return tensor.contiguous().to(torch::kCPU);
}

void dump_module_tensor(const std::string& layer_dump_dir,
                        const std::string& module_file,
                        const std::string& module_name,
                        const std::string& tensor_name,
                        const torch::Tensor& tensor) {
  if (layer_dump_dir.empty() || !tensor.defined()) {
    return;
  }
  const auto module_dir = layer_dump_dir + "/" +
                          sanitize_dump_name(module_file) + "/" +
                          sanitize_dump_name(module_name);
  try {
    std::filesystem::create_directories(module_dir);
  } catch (const std::filesystem::filesystem_error& e) {
    LOG(WARNING) << "[DSV4][Dump] failed to create " << module_dir << ": "
                 << e.what();
    return;
  }
  const auto path = module_dir + "/" + sanitize_dump_name(tensor_name) + ".pt";
  try {
    save_tensor_as_pickle(dump_tensor_on_cpu(tensor), path);
  } catch (const c10::Error& e) {
    LOG(WARNING) << "[DSV4][Dump] failed to save " << path << ": "
                 << e.what_without_backtrace();
  }
}

}  // namespace

DeepseekV4DecoderLayerImpl::DeepseekV4DecoderLayerImpl(
    const ModelContext& context,
    int32_t layer_id) {
  const auto& args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();
  layer_id_ = layer_id;
  tp_rank_ = parallel_args.tp_group_->rank();

  int64_t hidden_size = args.hidden_size();

  hc_mult_ = args.hc_mult();
  hc_sinkhorn_iters_ = args.hc_sinkhorn_iters();
  hc_eps_ = static_cast<double>(args.hc_eps());
  norm_eps_ = static_cast<double>(args.rms_norm_eps());

  attention_ = register_module("attn", DSAttention(context, layer_id));
  attn_norm_ = register_module(
      "attn_norm", RMSNorm(hidden_size, args.rms_norm_eps(), options));
  ffn_norm_ = register_module(
      "ffn_norm", RMSNorm(hidden_size, args.rms_norm_eps(), options));
  FusedMoEArgs moe_args;
  moe_args.is_gated = true;
  moe_args.debug_layer_id = layer_id;
  // DeepseekV4 drives expert routing through its own DeepseekV4Gate and only
  // calls forward_with_selected_experts().  The FusedMoE internal gate_ is
  // therefore never used; skip loading its weights to avoid redundant memory
  // allocation and a duplicate copy of the router weight matrix.
  moe_args.skip_gate_load = true;
  moe_mlp_ = register_module(
      "ffn", FusedMoE(args, moe_args, quant_args, parallel_args, options));
  // Register as "gate" to match Python's mlp.gate module path.
  gate_ = register_module("gate", DeepseekV4Gate(context, layer_id));

  const int64_t mix_hc = (2 + hc_mult_) * hc_mult_;
  const int64_t hc_dim = hc_mult_ * hidden_size;
  auto hc_options = options.dtype(torch::kFloat32);
  hc_attn_fn_ = register_parameter("hc_attn_fn",
                                   torch::empty({mix_hc, hc_dim}, hc_options),
                                   /*requires_grad=*/false);
  hc_ffn_fn_ = register_parameter("hc_ffn_fn",
                                  torch::empty({mix_hc, hc_dim}, hc_options),
                                  /*requires_grad=*/false);
  hc_attn_base_ = register_parameter("hc_attn_base",
                                     torch::empty({mix_hc}, hc_options),
                                     /*requires_grad=*/false);
  hc_ffn_base_ = register_parameter("hc_ffn_base",
                                    torch::empty({mix_hc}, hc_options),
                                    /*requires_grad=*/false);
  hc_attn_scale_ = register_parameter("hc_attn_scale",
                                      torch::empty({3}, hc_options),
                                      /*requires_grad=*/false);
  hc_ffn_scale_ = register_parameter("hc_ffn_scale",
                                     torch::empty({3}, hc_options),
                                     /*requires_grad=*/false);
}

void DeepseekV4DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  auto attn_state = state_dict.get_dict_with_prefix("attn.");
  if (attn_state.size() == 0) {
    attn_state = state_dict.get_dict_with_prefix("self_attn.");
  }
  if (attn_state.size() > 0) {
    attention_->load_state_dict(attn_state);
  }

  auto attn_norm_state = state_dict.get_dict_with_prefix("attn_norm.");
  if (attn_norm_state.size() == 0) {
    attn_norm_state = state_dict.get_dict_with_prefix("input_layernorm.");
  }
  if (attn_norm_state.size() > 0) {
    attn_norm_->load_state_dict(attn_norm_state);
  }

  auto ffn_norm_state = state_dict.get_dict_with_prefix("ffn_norm.");
  if (ffn_norm_state.size() == 0) {
    ffn_norm_state =
        state_dict.get_dict_with_prefix("post_attention_layernorm.");
  }
  if (ffn_norm_state.size() > 0) {
    ffn_norm_->load_state_dict(ffn_norm_state);
  }

  auto ffn_state = state_dict.get_dict_with_prefix("ffn.");
  if (ffn_state.size() == 0) {
    ffn_state = state_dict.get_dict_with_prefix("mlp.");
  }
  if (ffn_state.size() > 0) {
    auto gate_state = ffn_state.get_dict_with_prefix("gate.");
    if (gate_state.size() == 0) {
      gate_state = state_dict.get_dict_with_prefix("gate.");
    }
    if (gate_state.size() > 0) {
      gate_->load_state_dict(gate_state);
    }
    moe_mlp_->load_state_dict(ffn_state);
  }

  LOAD_WEIGHT(hc_attn_fn);
  LOAD_WEIGHT(hc_ffn_fn);
  LOAD_WEIGHT(hc_attn_base);
  LOAD_WEIGHT(hc_ffn_base);
  LOAD_WEIGHT(hc_attn_scale);
  LOAD_WEIGHT(hc_ffn_scale);
}

DeepseekV4LayerWeightMemStats DeepseekV4DecoderLayerImpl::get_weight_mem_stats()
    const {
  DeepseekV4LayerWeightMemStats stats;

  if (attention_) {
    stats.attn_bytes = module_registered_tensor_bytes(*attention_) +
                       attention_->non_registered_weight_bytes();
  }

  if (moe_mlp_) {
    stats.expert_bytes += module_registered_tensor_bytes(*moe_mlp_);
  }
  if (gate_) {
    stats.expert_bytes += module_registered_tensor_bytes(*gate_);
  }

  stats.hc_bytes = tensor_bytes(hc_attn_fn_) + tensor_bytes(hc_ffn_fn_) +
                   tensor_bytes(hc_attn_base_) + tensor_bytes(hc_ffn_base_) +
                   tensor_bytes(hc_attn_scale_) + tensor_bytes(hc_ffn_scale_);

  stats.total_bytes = module_registered_tensor_bytes(*this);
  if (attention_) {
    stats.total_bytes += attention_->non_registered_weight_bytes();
  }

  const int64_t categorized =
      stats.attn_bytes + stats.expert_bytes + stats.hc_bytes;
  stats.other_bytes = std::max<int64_t>(stats.total_bytes - categorized, 0);

  return stats;
}

torch::Tensor DeepseekV4DecoderLayerImpl::forward(
    torch::Tensor& x,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    const std::optional<torch::Tensor>& input_ids) {
  (void)positions;

  residual = std::nullopt;

  const int32_t dump_layer_id = deepseek_v4_dump_layer_id();
  const bool should_dump_target_layer = (layer_id_ == dump_layer_id);
  const auto layer_dump_dir = should_dump_target_layer
                                  ? make_layer_dump_dir(tp_rank_, dump_layer_id)
                                  : "";
  const auto log_moe_tensor = [&](const std::string& node,
                                  const torch::Tensor& tensor) {
    LOG(INFO) << "[DSV4][Node] file=deepseek_v4_decoder_layer.cpp layer="
              << layer_id_ << " module=moe tensor=" << node
              << " shape=" << tensor_shape_string(tensor)
              << " dtype_device=" << tensor_dtype_device_string(tensor);
  };
  const auto dump_moe_tensor = [&](const std::string& node,
                                   const torch::Tensor& tensor) {
    if (should_dump_target_layer) {
      dump_module_tensor(
          layer_dump_dir, "deepseek_v4_decoder_layer.cpp", "moe", node, tensor);
    }
  };
  const auto dump_vllm_tensor = [&](const std::string& node,
                                    const torch::Tensor& tensor) {
    moe_dump::dump_tensor(
        moe_dump::current_step(), tp_rank_, layer_id_, node, tensor);
  };

  CHECK(attn_metadata.dsa_metadata)
      << "DeepseekV4DecoderLayer requires DSA metadata for DSAttention path.";

  auto residual_attn = x;
  auto [attn_input, post_attn, comb_attn] =
      hc_pre(x, hc_attn_fn_, hc_attn_scale_, hc_attn_base_);
  attn_input = std::get<0>(attn_norm_->forward(attn_input));

  auto& dsa = *(attn_metadata.dsa_metadata);
  const auto compress_metadata = std::make_tuple(
      dsa.c1_metadata, dsa.c4_metadata, dsa.c128_metadata, dsa.qli_metadata);
  KVState kv_state{kv_cache.get_swa_cache(),
                   kv_cache.get_compress_kv_state(),
                   kv_cache.get_compress_score_state(),
                   kv_cache.get_compress_index_kv_state(),
                   kv_cache.get_compress_index_score_state()};
  auto [attn_output, attn_lse] = attention_->forward(
      dsa,
      attn_input,
      kv_cache,
      kv_state,
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill,
      std::to_string(dsa.layer_id),
      compress_metadata);
  (void)attn_lse;
  attn_input = attn_output;
  x = hc_post(attn_input, residual_attn, post_attn, comb_attn);

  auto residual_ffn = x;
  auto [ffn_input, post_ffn, comb_ffn] =
      hc_pre(x, hc_ffn_fn_, hc_ffn_scale_, hc_ffn_base_);
  ffn_input = std::get<0>(ffn_norm_->forward(ffn_input));
  dump_vllm_tensor("moe_input_hidden_states",
                   ffn_input.reshape({-1, ffn_input.size(-1)}));

  auto ffn_input_2d = ffn_input.reshape({-1, ffn_input.size(-1)});
  log_moe_tensor("ffn_input_2d.input", ffn_input_2d);
  dump_moe_tensor("ffn_input_2d.input", ffn_input_2d);
  std::optional<torch::Tensor> gate_input_ids = std::nullopt;
  if (input_ids.has_value() && input_ids.value().defined()) {
    auto flat_input_ids =
        input_ids.value().reshape({-1}).to(ffn_input.device());
    const int64_t token_count = flat_input_ids.size(0);
    const int64_t hidden_rows = ffn_input_2d.size(0);
    if (token_count == hidden_rows) {
      gate_input_ids = flat_input_ids;
    } else if (token_count > 0 && hidden_rows % token_count == 0) {
      const int64_t repeat_factor = hidden_rows / token_count;
      gate_input_ids = flat_input_ids.unsqueeze(1)
                           .repeat({1, repeat_factor})
                           .reshape({hidden_rows});
    }
  }
  if (gate_input_ids.has_value()) {
    log_moe_tensor("gate_input_ids.input", gate_input_ids.value());
    dump_moe_tensor("gate_input_ids.input", gate_input_ids.value());
    dump_vllm_tensor("gate_input_ids_after_comm", gate_input_ids.value());
  } else {
    log_moe_tensor("gate_input_ids.input", torch::Tensor());
  }
  if (gate_->is_hash_layer()) {
    CHECK(gate_input_ids.has_value())
        << "DeepseekV4 hash gate requires input_ids for routing";
  }
  auto [topk_weights, topk_ids] = gate_->forward(ffn_input_2d, gate_input_ids);
  dump_vllm_tensor("gate_router_logits", gate_->debug_last_router_logits());
  dump_vllm_tensor("gate_router_logits_before_hash_gating",
                   gate_->debug_last_router_logits_before_hash_gating());
  dump_vllm_tensor("gate_hash_topk_weights",
                   gate_->debug_last_hash_topk_weights());
  dump_vllm_tensor("gate_hash_topk_ids", gate_->debug_last_hash_topk_ids());
  dump_vllm_tensor("gate_weight", gate_->debug_weight());
  dump_vllm_tensor("gate_e_score_correction_bias",
                   gate_->debug_e_score_correction_bias());
  dump_vllm_tensor("gate_tid2eid", gate_->debug_tid2eid());
  dump_vllm_tensor("gate_hash_topk_weights", topk_weights);
  dump_vllm_tensor("gate_hash_topk_ids", topk_ids);
  log_moe_tensor("topk_weights.output", topk_weights);
  log_moe_tensor("topk_ids.output", topk_ids);
  dump_moe_tensor("topk_weights.output", topk_weights);
  dump_moe_tensor("topk_ids.output", topk_ids);
  ffn_input = moe_mlp_->forward_with_selected_experts(
      ffn_input, topk_weights, topk_ids, input_params);
  const auto& shared_output_pre = moe_mlp_->debug_last_shared_output_pre();
  const auto& shared_gate = moe_mlp_->debug_last_shared_gate();
  const auto& shared_output = moe_mlp_->debug_last_shared_output();
  dump_vllm_tensor("shared_experts_output", shared_output);
  dump_vllm_tensor("shared_fused_moe_shared_expert_out", shared_output);
  log_moe_tensor("shared_output_pre.output", shared_output_pre);
  log_moe_tensor("shared_gate.output", shared_gate);
  log_moe_tensor("shared_output.output", shared_output);
  dump_moe_tensor("shared_output_pre.output", shared_output_pre);
  dump_moe_tensor("shared_gate.output", shared_gate);
  dump_moe_tensor("shared_output.output", shared_output);
  log_moe_tensor("moe_output.output", ffn_input);
  dump_moe_tensor("moe_output.output", ffn_input);
  dump_vllm_tensor("routed_experts_output",
                   moe_mlp_->debug_last_routed_experts_output());
  dump_vllm_tensor("shared_fused_moe_routed_out",
                   moe_mlp_->debug_last_routed_experts_output());
  dump_vllm_tensor("shared_fused_moe_input_hidden_states",
                   moe_mlp_->debug_last_input_hidden_states());
  dump_vllm_tensor("shared_fused_moe_router_logits",
                   gate_->debug_last_router_logits());
  dump_vllm_tensor("moe_final_output_before_allreduce",
                   moe_mlp_->debug_last_before_allreduce());
  x = hc_post(ffn_input, residual_ffn, post_ffn, comb_ffn);

  return x;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
DeepseekV4DecoderLayerImpl::hc_pre(const torch::Tensor& x,
                                   const torch::Tensor& hc_fn,
                                   const torch::Tensor& hc_scale,
                                   const torch::Tensor& hc_base) {
  kernel::HcPreParams params;
  params.x = x;
  params.hc_fn = hc_fn;
  params.hc_scale = hc_scale;
  params.hc_base = hc_base;
  params.hc_mult = hc_mult_;
  params.hc_sinkhorn_iters = hc_sinkhorn_iters_;
  params.norm_eps = norm_eps_;
  params.hc_eps = hc_eps_;
  return kernel::hc_pre(params);
}

torch::Tensor DeepseekV4DecoderLayerImpl::hc_post(const torch::Tensor& x,
                                                  const torch::Tensor& residual,
                                                  const torch::Tensor& post,
                                                  const torch::Tensor& comb) {
  kernel::HcPostParams params;
  if (x.dim() == 2 && residual.dim() == 3 && post.dim() == 2 &&
      comb.dim() == 3) {
    params.x = x.unsqueeze(0);
    params.residual = residual.unsqueeze(0);
    params.post = post.unsqueeze(0);
    params.comb = comb.unsqueeze(0);
    return kernel::hc_post(params).squeeze(0);
  }

  params.x = x;
  params.residual = residual;
  params.post = post;
  params.comb = comb;
  return kernel::hc_post(params);
}

}  // namespace layer
}  // namespace xllm
