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
#include "deepseek_sparse_attention.h"

#include <glog/logging.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "kernels/ops_api.h"
#include "xllm/core/kernels/npu/xllm_ops/xllm_ops_api.h"
#include "xllm/core/util/tensor_helper.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {
namespace {

struct DsaCacheMapping {
  int64_t cmp_cache_idx = -1;
  int64_t index_cache_idx = -1;
  int64_t indexer_scale_cache_idx = -1;
  int64_t ori_cache_idx = -1;
  int64_t kv_state_cache_idx = -1;
  int64_t score_state_cache_idx = -1;
  int64_t index_kv_state_cache_idx = -1;
  int64_t index_score_state_cache_idx = -1;
};

c10::optional<torch::Tensor> as_optional(const torch::Tensor& tensor) {
  if (tensor.defined() && tensor.numel() > 0) {
    return c10::optional<torch::Tensor>(tensor);
  }
  return c10::nullopt;
}

torch::Tensor get_layer_cache_tensor(
    const std::vector<std::vector<torch::Tensor>>& layer_tensors,
    int32_t layer_id,
    int64_t cache_idx) {
  if (layer_id < 0 || layer_id >= static_cast<int32_t>(layer_tensors.size()) ||
      cache_idx < 0 ||
      cache_idx >= static_cast<int64_t>(layer_tensors[layer_id].size())) {
    return torch::Tensor();
  }
  return layer_tensors[layer_id][cache_idx];
}

DsaCacheMapping resolve_cache_mapping(const DSAMetadata& attn_metadata,
                                      int64_t compress_ratio) {
  DsaCacheMapping mapping;
  if (!attn_metadata.caches_info || attn_metadata.layer_id < 0 ||
      attn_metadata.layer_id >=
          static_cast<int32_t>(attn_metadata.caches_info->size())) {
    return mapping;
  }

  const auto& layer_caches =
      (*(attn_metadata.caches_info))[attn_metadata.layer_id];

  std::vector<int64_t> token_ratio_indices;
  std::vector<int64_t> swa_indices;
  token_ratio_indices.reserve(layer_caches.size());
  swa_indices.reserve(layer_caches.size());

  for (int64_t cache_idx = 0;
       cache_idx < static_cast<int64_t>(layer_caches.size());
       ++cache_idx) {
    const auto& cache_info = layer_caches[cache_idx];
    if (cache_info.type == DSACacheType::TOKEN &&
        cache_info.ratio == static_cast<int32_t>(compress_ratio)) {
      token_ratio_indices.push_back(cache_idx);
    }
    if (cache_info.type == DSACacheType::SLIDING_WINDOW) {
      swa_indices.push_back(cache_idx);
    }
  }

  if (!token_ratio_indices.empty() && compress_ratio > 1) {
    mapping.cmp_cache_idx = token_ratio_indices[0];
  }
  if (token_ratio_indices.size() > 1) {
    mapping.index_cache_idx = token_ratio_indices[1];
  }
  if (token_ratio_indices.size() > 2) {
    mapping.indexer_scale_cache_idx = token_ratio_indices[2];
  }

  if (!swa_indices.empty()) {
    mapping.ori_cache_idx = swa_indices[0];
  }
  if (swa_indices.size() > 1) {
    mapping.kv_state_cache_idx = swa_indices[1];
  }
  if (swa_indices.size() > 2) {
    mapping.score_state_cache_idx = swa_indices[2];
  }
  if (swa_indices.size() > 3) {
    mapping.index_kv_state_cache_idx = swa_indices[3];
  }
  if (swa_indices.size() > 4) {
    mapping.index_score_state_cache_idx = swa_indices[4];
  }

  return mapping;
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

std::string rope_dump_root() {
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

torch::Tensor dump_tensor_on_cpu(const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return torch::Tensor();
  }
  return tensor.contiguous().to(torch::kCPU);
}

void dump_rope_tensor(const std::string& layer_dir,
                      const std::string& name,
                      const torch::Tensor& tensor) {
  if (layer_dir.empty() || !tensor.defined()) {
    return;
  }
  const auto path = layer_dir + "/" + sanitize_dump_name(name) + ".pt";
  torch::Tensor out;
  try {
    out = dump_tensor_on_cpu(tensor);
  } catch (const c10::Error& e) {
    LOG(WARNING) << "[DSV4][RoPE Dump] failed to prepare " << path << ": "
                 << e.what_without_backtrace();
    return;
  }
  try {
    save_tensor_as_pickle(out, path);
  } catch (const c10::Error& e) {
    LOG(WARNING) << "[DSV4][RoPE Dump] failed to save " << path << ": "
                 << e.what_without_backtrace();
  }
}

void dump_rope_module_tensor(const std::string& layer_dir,
                             const std::string& module_name,
                             const std::string& tensor_name,
                             const torch::Tensor& tensor) {
  if (layer_dir.empty() || module_name.empty() || !tensor.defined()) {
    return;
  }
  const auto module_dir = layer_dir + "/" + sanitize_dump_name(module_name);
  try {
    std::filesystem::create_directories(module_dir);
  } catch (const std::filesystem::filesystem_error& e) {
    LOG(WARNING) << "[DSV4][RoPE Dump] failed to create " << module_dir << ": "
                 << e.what();
    return;
  }
  const auto path = module_dir + "/" + sanitize_dump_name(tensor_name) + ".pt";
  torch::Tensor out;
  try {
    out = dump_tensor_on_cpu(tensor);
  } catch (const c10::Error& e) {
    LOG(WARNING) << "[DSV4][RoPE Dump] failed to prepare " << path << ": "
                 << e.what_without_backtrace();
    return;
  }
  try {
    save_tensor_as_pickle(out, path);
  } catch (const c10::Error& e) {
    LOG(WARNING) << "[DSV4][RoPE Dump] failed to save " << path << ": "
                 << e.what_without_backtrace();
  }
}

void dump_rope_module_text(const std::string& layer_dir,
                           const std::string& module_name,
                           const std::string& name,
                           const std::string& text) {
  if (layer_dir.empty() || module_name.empty()) {
    return;
  }
  const auto module_dir = layer_dir + "/" + sanitize_dump_name(module_name);
  try {
    std::filesystem::create_directories(module_dir);
  } catch (const std::filesystem::filesystem_error& e) {
    LOG(WARNING) << "[DSV4][RoPE Dump] failed to create " << module_dir << ": "
                 << e.what();
    return;
  }
  const auto path = module_dir + "/" + sanitize_dump_name(name) + ".txt";
  std::ofstream ofs(path, std::ios::out | std::ios::trunc);
  if (!ofs) {
    LOG(WARNING) << "[DSV4][RoPE Dump] failed to open " << path;
    return;
  }
  ofs << text;
}

void dump_rope_text(const std::string& layer_dir,
                    const std::string& name,
                    const std::string& text) {
  if (layer_dir.empty()) {
    return;
  }
  const auto path = layer_dir + "/" + sanitize_dump_name(name) + ".txt";
  std::ofstream ofs(path, std::ios::out | std::ios::trunc);
  if (!ofs) {
    LOG(WARNING) << "[DSV4][RoPE Dump] failed to open " << path;
    return;
  }
  ofs << text;
}

std::string make_rope_layer_dump_dir(int64_t tp_rank, int32_t layer_id) {
  auto root = rope_dump_root();
  if (root.empty()) {
    return "";
  }
  const auto tp_rank_name = "tp_rank_" + std::to_string(tp_rank);
  const auto layer_name = "layer_" + std::to_string(layer_id);
  const auto layer_dir = root + "/" + tp_rank_name + "/" + layer_name;
  try {
    std::filesystem::create_directories(layer_dir);
  } catch (const std::filesystem::filesystem_error& e) {
    LOG(WARNING) << "[DSV4][RoPE Dump] failed to create " << layer_dir << ": "
                 << e.what();
    return "";
  }
  return layer_dir;
}

void dump_rope_call(const std::string& layer_dir,
                    const std::string& stage,
                    int32_t layer_id,
                    int64_t rope_start_dim,
                    int64_t rope_head_dim,
                    bool inverse,
                    const torch::Tensor& input_before,
                    const torch::Tensor& cos,
                    const torch::Tensor& sin,
                    const torch::Tensor& input_after) {
  LOG(INFO) << "[DSV4][RoPE] layer=" << layer_id << " stage=" << stage
            << " inverse=" << inverse << " rope_start_dim=" << rope_start_dim
            << " rope_head_dim=" << rope_head_dim
            << " input=" << tensor_shape_string(input_before) << "/"
            << tensor_dtype_device_string(input_before)
            << " cos=" << tensor_shape_string(cos) << "/"
            << tensor_dtype_device_string(cos)
            << " sin=" << tensor_shape_string(sin) << "/"
            << tensor_dtype_device_string(sin)
            << " output=" << tensor_shape_string(input_after) << "/"
            << tensor_dtype_device_string(input_after);

  if (layer_dir.empty()) {
    return;
  }
  const auto safe_stage = sanitize_dump_name(stage);
  const auto rope_module_name = "rope_" + safe_stage;
  std::ostringstream meta;
  meta << "layer=" << layer_id << "\n"
       << "stage=" << stage << "\n"
       << "inverse=" << inverse << "\n"
       << "rope_start_dim=" << rope_start_dim << "\n"
       << "rope_head_dim=" << rope_head_dim << "\n"
       << "input_shape=" << tensor_shape_string(input_before) << "\n"
       << "input_dtype_device=" << tensor_dtype_device_string(input_before)
       << "\n"
       << "cos_shape=" << tensor_shape_string(cos) << "\n"
       << "cos_dtype_device=" << tensor_dtype_device_string(cos) << "\n"
       << "sin_shape=" << tensor_shape_string(sin) << "\n"
       << "sin_dtype_device=" << tensor_dtype_device_string(sin) << "\n"
       << "output_shape=" << tensor_shape_string(input_after) << "\n"
       << "output_dtype_device=" << tensor_dtype_device_string(input_after)
       << "\n";
  dump_rope_module_text(layer_dir, rope_module_name, "meta", meta.str());
  dump_rope_module_tensor(
      layer_dir, rope_module_name, "input_before", input_before);
  dump_rope_module_tensor(layer_dir, rope_module_name, "cos", cos);
  dump_rope_module_tensor(layer_dir, rope_module_name, "sin", sin);
  dump_rope_module_tensor(
      layer_dir, rope_module_name, "input_after", input_after);
}

void apply_partial_rope(torch::Tensor& input,
                        int64_t rope_start_dim,
                        int64_t rope_head_dim,
                        const torch::Tensor& cos,
                        const torch::Tensor& sin,
                        bool inverse = false,
                        int32_t layer_id = -1,
                        const std::string& stage = "rope",
                        const std::string& layer_dump_dir = "") {
  if (!input.defined() || !cos.defined() || !sin.defined() ||
      rope_head_dim <= 0 || rope_start_dim < 0) {
    return;
  }

  const int64_t input_last_dim = input.size(input.dim() - 1);
  if (input_last_dim < rope_start_dim + rope_head_dim) {
    return;
  }

  auto sin_cache = inverse ? -sin : sin;
  auto cos_cache = cos;
  CHECK(input.dim() == 2 || input.dim() == 3)
      << "apply_partial_rope only supports input dim 2/3, got: " << input.dim();
  CHECK(cos_cache.dim() == 2 && sin_cache.dim() == 2)
      << "apply_partial_rope expects cos/sin dim=2, got cos dim "
      << cos_cache.dim() << ", sin dim " << sin_cache.dim();
  CHECK(cos_cache.size(0) == input.size(0) &&
        sin_cache.size(0) == input.size(0))
      << "apply_partial_rope expects cos/sin batch == input.size(0), got cos "
      << cos_cache.size(0) << ", sin " << sin_cache.size(0) << ", input "
      << input.size(0);
  CHECK(cos_cache.size(1) == rope_head_dim &&
        sin_cache.size(1) == rope_head_dim)
      << "apply_partial_rope expects cos/sin last dim == rope_head_dim("
      << rope_head_dim << "), got cos " << cos_cache.size(1) << ", sin "
      << sin_cache.size(1);

  auto input_before = input.detach().clone();
  auto cos_4d = cos_cache.view({cos_cache.size(0), 1, 1, rope_head_dim});
  auto sin_4d = sin_cache.view({sin_cache.size(0), 1, 1, rope_head_dim});
  auto input_4d =
      (input.dim() == 3) ? input.unsqueeze(2) : input.unsqueeze(1).unsqueeze(1);
  xllm::kernel::NpuInplacePartialRotaryMulParams rope_params;
  rope_params.x = input_4d;
  rope_params.r1 = cos_4d;
  rope_params.r2 = sin_4d;
  rope_params.rotary_mode = "interleave";
  rope_params.partial_slice = {rope_start_dim, rope_start_dim + rope_head_dim};
  xllm::kernel::npu_inplace_partial_rotary_mul(rope_params);
  input =
      (input.dim() == 3) ? input_4d.squeeze(2) : input_4d.squeeze(1).squeeze(1);
  dump_rope_call(layer_dump_dir,
                 stage,
                 layer_id,
                 rope_start_dim,
                 rope_head_dim,
                 inverse,
                 input_before,
                 cos,
                 sin_cache,
                 input);
}

void scatter_by_slot(torch::Tensor& cache,
                     const torch::Tensor& slot_mapping,
                     const torch::Tensor& value) {
  if (!cache.defined() || !slot_mapping.defined() || !value.defined()) {
    return;
  }
  if (slot_mapping.numel() == 0 || value.numel() == 0) {
    return;
  }

  auto value_2d = value.reshape({-1, value.size(value.dim() - 1)});
  auto cache_2d = cache.view({-1, value_2d.size(1)});

  auto slots = slot_mapping.reshape({-1}).to(torch::kLong).to(cache.device());
  const int64_t update_rows =
      std::min<int64_t>(slots.size(0), value_2d.size(0));
  if (update_rows <= 0) {
    return;
  }

  auto slots_slice = slots.slice(/*dim=*/0, /*start=*/0, /*end=*/update_rows);
  auto value_slice =
      value_2d.slice(/*dim=*/0, /*start=*/0, /*end=*/update_rows);

  const int64_t cache_rows = cache_2d.size(0);
  CHECK_GT(cache_rows, 0) << "scatter_by_slot requires cache rows > 0, cache "
                          << cache.sizes();
  const int64_t min_slot = slots_slice.min().item<int64_t>();
  const int64_t max_slot = slots_slice.max().item<int64_t>();
  CHECK_GE(min_slot, 0) << "scatter_by_slot found negative slot index "
                        << min_slot << ", cache_rows=" << cache_rows
                        << ", slot_mapping_shape=" << slot_mapping.sizes()
                        << ", value_shape=" << value.sizes()
                        << ", cache_shape=" << cache.sizes();
  CHECK_LT(max_slot, cache_rows)
      << "scatter_by_slot slot index out of range: max_slot=" << max_slot
      << ", cache_rows=" << cache_rows
      << ", slot_mapping_shape=" << slot_mapping.sizes()
      << ", value_shape=" << value.sizes()
      << ", value_rows=" << value_2d.size(0) << ", update_rows=" << update_rows
      << ", cache_shape=" << cache.sizes();
  cache_2d.index_copy_(/*dim=*/0, slots_slice, value_slice);
}

int64_t tensor_max_or_zero(const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return 0;
  }
  return tensor.max().item<int64_t>();
}

AttentionMetadata build_indexer_attention_metadata(
    const DSAMetadata& attn_metadata,
    const torch::Tensor& block_table,
    const torch::Tensor& slot_mapping,
    bool is_prefill) {
  AttentionMetadata metadata;
  metadata.is_prefill = is_prefill;
  metadata.is_chunked_prefill = false;
  metadata.is_dummy = false;
  metadata.is_causal = true;

  metadata.block_table = block_table;
  metadata.slot_mapping = slot_mapping;

  metadata.q_cu_seq_lens = attn_metadata.actual_seq_lengths_query;
  metadata.kv_seq_lens = attn_metadata.actual_seq_lengths_kv;
  if (!metadata.kv_seq_lens.defined()) {
    metadata.kv_seq_lens = attn_metadata.seq_lens;
  }

  if (attn_metadata.actual_seq_lengths_query.defined() &&
      attn_metadata.actual_seq_lengths_query.dim() > 0 &&
      attn_metadata.actual_seq_lengths_query.size(0) > 1) {
    metadata.q_seq_lens = attn_metadata.actual_seq_lengths_query.slice(
        /*dim=*/0,
        /*start=*/1,
        /*end=*/attn_metadata.actual_seq_lengths_query.size(0));
  } else {
    metadata.q_seq_lens = attn_metadata.seq_lens_q;
  }

  if (metadata.kv_seq_lens.defined()) {
    auto kv_seq_lens = metadata.kv_seq_lens.to(torch::kInt32);
    auto kv_cumsum = torch::cumsum(kv_seq_lens, /*dim=*/0);
    metadata.kv_cu_seq_lens =
        torch::cat({torch::zeros({1}, kv_seq_lens.options()), kv_cumsum});
  }

  if (attn_metadata.max_seqlen_q.defined() &&
      attn_metadata.max_seqlen_q.numel() > 0) {
    metadata.max_query_len = attn_metadata.max_seqlen_q.max().item<int64_t>();
  } else {
    metadata.max_query_len = tensor_max_or_zero(metadata.q_seq_lens);
  }

  if (attn_metadata.max_seqlen_kv.defined() &&
      attn_metadata.max_seqlen_kv.numel() > 0) {
    metadata.max_seq_len = attn_metadata.max_seqlen_kv.max().item<int64_t>();
  } else {
    metadata.max_seq_len = tensor_max_or_zero(metadata.kv_seq_lens);
  }

  metadata.dsa_metadata = std::make_shared<DSAMetadata>(attn_metadata);

  return metadata;
}

}  // namespace

DSAttentionImpl::DSAttentionImpl(const ModelContext& context, int32_t layer_id)
    : DSAttentionImpl(context.get_model_args(),
                      context.get_quant_args(),
                      context.get_parallel_args(),
                      context.get_tensor_options(),
                      layer_id) {}

DSAttentionImpl::DSAttentionImpl(const ModelArgs& args,
                                 const QuantArgs& quant_args,
                                 const ParallelArgs& parallel_args,
                                 const torch::TensorOptions& options,
                                 int32_t layer_id)
    : num_heads_(args.n_heads()),
      head_size_(args.head_dim()),
      head_dim_(args.head_dim()),
      n_kv_heads_(args.n_kv_heads().value()),
      sliding_window_(-1),
      q_lora_rank_(args.q_lora_rank()),
      o_lora_rank_(args.o_lora_rank()),
      o_groups_(args.o_groups()),
      rope_head_dim_(args.rope_head_dim()),
      window_size_(args.window_size()),
      compress_ratio_(1.0),
      index_n_heads_(args.index_n_heads()),
      index_head_dim_(args.index_head_dim()),
      index_topk_(args.index_topk()),
      eps_(args.rms_norm_eps()) {
  const auto& compress_ratios = args.compress_ratios();
  CHECK(!compress_ratios.empty())
      << "DSAttention requires non-empty compress_ratios for DeepSeek V4";
  CHECK_GE(layer_id, 0) << "DSAttention requires valid layer_id, got "
                        << layer_id;
  CHECK_LT(layer_id, static_cast<int32_t>(compress_ratios.size()))
      << "DSAttention layer_id " << layer_id << " exceeds compress_ratios size "
      << compress_ratios.size();
  int64_t compress_ratio = compress_ratios[static_cast<size_t>(layer_id)];

  CHECK(compress_ratio == 1 || compress_ratio == 4 || compress_ratio == 128)
      << "DSAttention unsupported compress_ratio " << compress_ratio
      << " at layer " << layer_id;

  compress_ratio_ = static_cast<double>(compress_ratio);

  softmax_scale_ = std::pow(head_dim_, static_cast<double>(-0.5));
  scale_ = static_cast<float>(softmax_scale_);
  nope_head_dim_ = head_dim_ - rope_head_dim_;
  qk_head_dim_ = nope_head_dim_ + rope_head_dim_;

  LOG(INFO) << "[DSV4][HeadDim][AttentionInit] layer=" << layer_id
            << " attention_head_dim(args.head_dim)=" << head_dim_
            << " head_size=" << head_size_
            << " rope_head_dim=" << rope_head_dim_
            << " nope_head_dim=" << nope_head_dim_
            << " qk_head_dim=" << qk_head_dim_
            << " q_lora_rank=" << q_lora_rank_
            << " o_lora_rank=" << o_lora_rank_
            << " compress_ratio=" << compress_ratio_;

  const int64_t tp_size = parallel_args.tp_group_->world_size();
  tp_rank_ = parallel_args.tp_group_->rank();
  tp_size_ = tp_size;
  int64_t hidden_size = args.hidden_size();
  int64_t num_heads = args.n_heads();

  CHECK_EQ(o_groups_ % tp_size, 0)
      << "o_groups must be divisible by tensor parallel size";
  CHECK_EQ(num_heads % tp_size, 0)
      << "num_heads must be divisible by tensor parallel size";
  n_local_heads_ = num_heads / tp_size;
  n_local_groups_ = o_groups_ / tp_size;

  attn_sink_ = register_parameter(
      "attn_sink",
      torch::zeros({n_local_heads_}, options.dtype(torch::kFloat32)),
      /*requires_grad=*/false);

  q_a_proj_ = register_module(
      "q_a_proj",
      ReplicatedLinear(hidden_size, q_lora_rank_, false, QuantArgs(), options));

  q_layernorm_ =
      register_module("q_a_layernorm", RMSNorm(q_lora_rank_, eps_, options));

  q_b_proj_ = register_module("q_b_proj",
                              ColumnParallelLinear(q_lora_rank_,
                                                   num_heads * head_dim_,
                                                   false,
                                                   false,
                                                   quant_args,
                                                   parallel_args.tp_group_,
                                                   options));

  kv_proj_ = register_module(
      "kv_proj",
      ReplicatedLinear(hidden_size, head_dim_, false, QuantArgs(), options));
  kv_layernorm_ =
      register_module("kv_layernorm", RMSNorm(head_dim_, eps_, options));

  if (compress_ratio_ > 1) {
    compressor_ =
        register_module("compressor",
                        Compressor(static_cast<int64_t>(compress_ratio_),
                                   head_dim_,
                                   rope_head_dim_,
                                   /*rot_mode=*/2,
                                   eps_,
                                   options));
  }

  if (compress_ratio_ == 4) {
    if (index_n_heads_ <= 0) {
      index_n_heads_ = num_heads_;
    }
    if (index_head_dim_ <= 0) {
      index_head_dim_ = head_dim_;
    }
    if (index_topk_ <= 0) {
      index_topk_ = 512;
    }

    if (index_n_heads_ > 0 && index_head_dim_ > 0 && index_topk_ > 0) {
      indexer_ = register_module(
          "indexer",
          DeepseekV4Indexer(hidden_size,
                            index_n_heads_,
                            index_head_dim_,
                            rope_head_dim_,
                            index_topk_,
                            q_lora_rank_,
                            static_cast<int64_t>(compress_ratio_),
                            eps_,
                            quant_args,
                            options));
    } else {
      LOG(FATAL) << "DSAttention indexer disabled due to invalid config: "
                 << "index_n_heads=" << index_n_heads_
                 << ", index_head_dim=" << index_head_dim_
                 << ", index_topk=" << index_topk_;
    }
  }

  q_rms_gamma_ = register_buffer("q_rms_gamma",
                                 torch::ones({head_dim_},
                                             torch::TensorOptions()
                                                 .dtype(options.dtype())
                                                 .device(options.device())));

  o_a_proj_ =
      register_module("o_a_proj",
                      ColumnParallelLinear(num_heads * head_dim_ / o_groups_,
                                           o_groups_ * o_lora_rank_,
                                           false,
                                           true,
                                           quant_args,
                                           parallel_args.tp_group_,
                                           options));

  o_b_proj_ = register_module("o_b_proj",
                              RowParallelLinear(o_groups_ * o_lora_rank_,
                                                hidden_size,
                                                false,
                                                true,
                                                /*reduce=*/false,
                                                quant_args,
                                                parallel_args.tp_group_,
                                                options));
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
DSAttentionImpl::forward(const DSAMetadata& attn_metadata,
                         torch::Tensor& hidden_states,
                         KVCache& kv_cache,
                         KVState& kv_state,
                         bool isprefill,
                         std::string layer_name,
                         const std::tuple<torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor>& compress_metadata) {
  (void)layer_name;

  auto [c1_metadata, c4_metadata, c128_metadata, qli_metadata] =
      compress_metadata;
  const int32_t layer_id = attn_metadata.layer_id;
  const bool should_dump_layer0 = (layer_id == 0);
  const auto rope_layer_dump_dir =
      should_dump_layer0 ? make_rope_layer_dump_dir(tp_rank_, layer_id) : "";
  LOG(INFO) << "[DSV4][RoPE Dump] tp_rank=" << tp_rank_
            << " tp_size=" << tp_size_ << " layer=" << layer_id
            << " enabled=" << should_dump_layer0
            << " dump_dir=" << rope_layer_dump_dir;

  const auto log_node_tensor = [&](const std::string& node,
                                   const torch::Tensor& tensor) {
    LOG(INFO) << "[DSV4][Node] layer=" << layer_id << " node=" << node
              << " shape=" << tensor_shape_string(tensor)
              << " dtype_device=" << tensor_dtype_device_string(tensor);
  };
  const auto dump_node_tensor = [&](const std::string& node,
                                    const torch::Tensor& tensor) {
    if (should_dump_layer0 && !rope_layer_dump_dir.empty()) {
      dump_rope_tensor(rope_layer_dump_dir, node, tensor);
    }
  };
  const auto dump_module_tensor = [&](const std::string& module_name,
                                      const std::string& node,
                                      const torch::Tensor& tensor) {
    if (should_dump_layer0 && !rope_layer_dump_dir.empty()) {
      dump_rope_module_tensor(rope_layer_dump_dir, module_name, node, tensor);
    }
  };

  LOG(INFO) << "[DSV4][HeadDim][AttentionForward] layer=" << layer_id
            << " attention_head_dim=" << head_dim_
            << " rope_head_dim=" << rope_head_dim_
            << " nope_head_dim=" << nope_head_dim_
            << " qk_head_dim=" << qk_head_dim_
            << " hidden_states=" << tensor_shape_string(hidden_states)
            << " cos=" << tensor_shape_string(attn_metadata.cos)
            << " sin=" << tensor_shape_string(attn_metadata.sin);
  log_node_tensor("hidden_states.input", hidden_states);
  log_node_tensor("cos.input", attn_metadata.cos);
  log_node_tensor("sin.input", attn_metadata.sin);
  dump_node_tensor("hidden_states.input", hidden_states);
  dump_node_tensor("cos.input", attn_metadata.cos);
  dump_node_tensor("sin.input", attn_metadata.sin);

  // 1) q projection + q rmsnorm
  auto q_down = q_a_proj_->forward(hidden_states);
  log_node_tensor("q_a_proj.output", q_down);
  dump_node_tensor("q_a_proj.output", q_down);
  auto qr = std::get<0>(q_layernorm_->forward(q_down));
  log_node_tensor("q_layernorm.output", qr);
  dump_node_tensor("q_layernorm.output", qr);
  auto q = q_b_proj_->forward(qr).view({-1, n_local_heads_, head_dim_});
  log_node_tensor("q_b_proj_reshape.output", q);
  dump_node_tensor("q_b_proj_reshape.output", q);

  xllm::kernel::FusedLayerNormParams q_rmsnorm_params;
  q_rmsnorm_params.input = q;
  q_rmsnorm_params.weight = q_rms_gamma_;
  q_rmsnorm_params.mode = "rmsnorm";
  q_rmsnorm_params.eps = eps_;
  xllm::kernel::fused_layernorm(q_rmsnorm_params);
  q = q_rmsnorm_params.output;
  log_node_tensor("q_rmsnorm.output", q);
  dump_node_tensor("q_rmsnorm.output", q);

  // 2) kv projection
  auto kv_down = kv_proj_->forward(hidden_states);
  log_node_tensor("kv_proj.output", kv_down);
  dump_node_tensor("kv_proj.output", kv_down);
  auto kv = std::get<0>(kv_layernorm_->forward(kv_down));
  log_node_tensor("kv_layernorm.output", kv);
  dump_node_tensor("kv_layernorm.output", kv);
  kv = kv.view({-1, 1, qk_head_dim_});
  log_node_tensor("kv_reshape.output", kv);
  dump_node_tensor("kv_reshape.output", kv);

  // 3) RoPE (q and kv)
  auto cos = attn_metadata.cos;
  auto sin = attn_metadata.sin;
  apply_partial_rope(q,
                     nope_head_dim_,
                     rope_head_dim_,
                     cos,
                     sin,
                     /*inverse=*/false,
                     layer_id,
                     "q_forward_rope",
                     rope_layer_dump_dir);
  log_node_tensor("q_after_rope.output", q);
  dump_node_tensor("q_after_rope.output", q);
  apply_partial_rope(kv,
                     nope_head_dim_,
                     rope_head_dim_,
                     cos,
                     sin,
                     /*inverse=*/false,
                     layer_id,
                     "kv_forward_rope",
                     rope_layer_dump_dir);
  log_node_tensor("kv_after_rope.output", kv);
  dump_node_tensor("kv_after_rope.output", kv);

  // 4) resolve per-layer cache mapping
  const int64_t compress_ratio_i = static_cast<int64_t>(compress_ratio_);
  DsaCacheMapping mapping =
      resolve_cache_mapping(attn_metadata, compress_ratio_i);
  LOG(INFO) << "[DSV4][DSA][CacheMapping] layer=" << layer_id
            << " compress_ratio=" << compress_ratio_i
            << " cmp_cache_idx=" << mapping.cmp_cache_idx
            << " index_cache_idx=" << mapping.index_cache_idx
            << " indexer_scale_cache_idx=" << mapping.indexer_scale_cache_idx
            << " ori_cache_idx=" << mapping.ori_cache_idx
            << " kv_state_cache_idx=" << mapping.kv_state_cache_idx
            << " score_state_cache_idx=" << mapping.score_state_cache_idx
            << " index_kv_state_cache_idx=" << mapping.index_kv_state_cache_idx
            << " index_score_state_cache_idx="
            << mapping.index_score_state_cache_idx;

  auto cmp_block_table = get_layer_cache_tensor(attn_metadata.block_tables,
                                                attn_metadata.layer_id,
                                                mapping.cmp_cache_idx);
  auto ori_block_table = get_layer_cache_tensor(attn_metadata.block_tables,
                                                attn_metadata.layer_id,
                                                mapping.ori_cache_idx);
  auto kv_block_table = get_layer_cache_tensor(attn_metadata.block_tables,
                                               attn_metadata.layer_id,
                                               mapping.kv_state_cache_idx);
  auto score_block_table =
      get_layer_cache_tensor(attn_metadata.block_tables,
                             attn_metadata.layer_id,
                             mapping.score_state_cache_idx);
  auto index_kv_block_table =
      get_layer_cache_tensor(attn_metadata.block_tables,
                             attn_metadata.layer_id,
                             mapping.index_kv_state_cache_idx);
  auto index_score_block_table =
      get_layer_cache_tensor(attn_metadata.block_tables,
                             attn_metadata.layer_id,
                             mapping.index_score_state_cache_idx);
  auto index_block_table = get_layer_cache_tensor(attn_metadata.block_tables,
                                                  attn_metadata.layer_id,
                                                  mapping.index_cache_idx);

  auto cmp_slot = get_layer_cache_tensor(attn_metadata.slot_mappings,
                                         attn_metadata.layer_id,
                                         mapping.cmp_cache_idx);
  auto ori_slot = get_layer_cache_tensor(attn_metadata.slot_mappings,
                                         attn_metadata.layer_id,
                                         mapping.ori_cache_idx);
  auto index_slot = get_layer_cache_tensor(attn_metadata.slot_mappings,
                                           attn_metadata.layer_id,
                                           mapping.index_cache_idx);
  LOG(INFO) << "[DSV4][DSA][LayerInputs] layer=" << layer_id
            << " cmp_block_table=" << tensor_shape_string(cmp_block_table)
            << "/" << tensor_dtype_device_string(cmp_block_table)
            << " ori_block_table=" << tensor_shape_string(ori_block_table)
            << "/" << tensor_dtype_device_string(ori_block_table)
            << " kv_block_table=" << tensor_shape_string(kv_block_table) << "/"
            << tensor_dtype_device_string(kv_block_table)
            << " score_block_table=" << tensor_shape_string(score_block_table)
            << "/" << tensor_dtype_device_string(score_block_table)
            << " index_kv_block_table="
            << tensor_shape_string(index_kv_block_table) << "/"
            << tensor_dtype_device_string(index_kv_block_table)
            << " index_score_block_table="
            << tensor_shape_string(index_score_block_table) << "/"
            << tensor_dtype_device_string(index_score_block_table)
            << " index_block_table=" << tensor_shape_string(index_block_table)
            << "/" << tensor_dtype_device_string(index_block_table)
            << " cmp_slot=" << tensor_shape_string(cmp_slot) << "/"
            << tensor_dtype_device_string(cmp_slot)
            << " ori_slot=" << tensor_shape_string(ori_slot) << "/"
            << tensor_dtype_device_string(ori_slot)
            << " index_slot=" << tensor_shape_string(index_slot) << "/"
            << tensor_dtype_device_string(index_slot);
  dump_module_tensor("dsa_layer_inputs", "cmp_block_table", cmp_block_table);
  dump_module_tensor("dsa_layer_inputs", "ori_block_table", ori_block_table);
  dump_module_tensor("dsa_layer_inputs", "kv_block_table", kv_block_table);
  dump_module_tensor(
      "dsa_layer_inputs", "score_block_table", score_block_table);
  dump_module_tensor(
      "dsa_layer_inputs", "index_kv_block_table", index_kv_block_table);
  dump_module_tensor(
      "dsa_layer_inputs", "index_score_block_table", index_score_block_table);
  dump_module_tensor(
      "dsa_layer_inputs", "index_block_table", index_block_table);
  dump_module_tensor("dsa_layer_inputs", "cmp_slot", cmp_slot);
  dump_module_tensor("dsa_layer_inputs", "ori_slot", ori_slot);
  dump_module_tensor("dsa_layer_inputs", "index_slot", index_slot);

  auto ori_kv = std::get<0>(kv_state);
  if (!ori_kv.defined()) {
    ori_kv = kv_cache.get_swa_cache();
  }

  auto compressor_kv_state = std::get<1>(kv_state);
  if (!compressor_kv_state.defined()) {
    compressor_kv_state = kv_cache.get_compress_kv_state();
  }

  auto compressor_score_state = std::get<2>(kv_state);
  if (!compressor_score_state.defined()) {
    compressor_score_state = kv_cache.get_compress_score_state();
  }

  auto index_kv_state = std::get<3>(kv_state);
  if (!index_kv_state.defined()) {
    index_kv_state = kv_cache.get_compress_index_kv_state();
  }

  auto index_score_state = std::get<4>(kv_state);
  if (!index_score_state.defined()) {
    index_score_state = kv_cache.get_compress_index_score_state();
  }
  LOG(INFO) << "[DSV4][DSA][KVState] layer=" << layer_id
            << " ori_kv=" << tensor_shape_string(ori_kv) << "/"
            << tensor_dtype_device_string(ori_kv) << " compressor_kv_state="
            << tensor_shape_string(compressor_kv_state) << "/"
            << tensor_dtype_device_string(compressor_kv_state)
            << " compressor_score_state="
            << tensor_shape_string(compressor_score_state) << "/"
            << tensor_dtype_device_string(compressor_score_state)
            << " index_kv_state=" << tensor_shape_string(index_kv_state) << "/"
            << tensor_dtype_device_string(index_kv_state)
            << " index_score_state=" << tensor_shape_string(index_score_state)
            << "/" << tensor_dtype_device_string(index_score_state);
  dump_module_tensor("dsa_kv_state", "ori_kv", ori_kv);
  dump_module_tensor(
      "dsa_kv_state", "compressor_kv_state", compressor_kv_state);
  dump_module_tensor(
      "dsa_kv_state", "compressor_score_state", compressor_score_state);
  dump_module_tensor("dsa_kv_state", "index_kv_state", index_kv_state);
  dump_module_tensor("dsa_kv_state", "index_score_state", index_score_state);

  // 5) write ori kv cache
  scatter_by_slot(ori_kv, ori_slot, kv);

  // 6) optional compressor for cmp cache
  auto cmp_kv = kv_cache.get_k_cache();
  if (compress_ratio_i > 1 && compressor_ && cmp_kv.defined() &&
      cmp_slot.defined() && compressor_kv_state.defined() &&
      compressor_score_state.defined()) {
    torch::Tensor compress_cos;
    torch::Tensor compress_sin;
    if (compress_ratio_i == 4) {
      compress_cos = attn_metadata.c4_cos;
      compress_sin = attn_metadata.c4_sin;
    } else if (compress_ratio_i == 128) {
      compress_cos = attn_metadata.c128_cos;
      compress_sin = attn_metadata.c128_sin;
    }

    LOG(INFO) << "[DSV4][RoPE][CompressorInputs] layer=" << layer_id
              << " compress_ratio=" << compress_ratio_i
              << " hidden_states=" << tensor_shape_string(hidden_states)
              << " compress_sin=" << tensor_shape_string(compress_sin) << "/"
              << tensor_dtype_device_string(compress_sin)
              << " compress_cos=" << tensor_shape_string(compress_cos) << "/"
              << tensor_dtype_device_string(compress_cos)
              << " actual_seq_lengths_query="
              << tensor_shape_string(attn_metadata.actual_seq_lengths_query);
    dump_rope_tensor(
        rope_layer_dump_dir, "compressor.hidden_states.input", hidden_states);
    dump_rope_tensor(
        rope_layer_dump_dir, "compressor.compress_sin.input", compress_sin);
    dump_rope_tensor(
        rope_layer_dump_dir, "compressor.compress_cos.input", compress_cos);
    dump_rope_tensor(rope_layer_dump_dir,
                     "compressor.actual_seq_lengths_query.input",
                     attn_metadata.actual_seq_lengths_query);

    std::tuple<torch::Tensor, torch::Tensor> compressor_states{
        compressor_kv_state, compressor_score_state};
    std::tuple<torch::Tensor, torch::Tensor> compressor_block_tables{
        kv_block_table, score_block_table};

    auto compressed_kv =
        compressor_->forward(attn_metadata,
                             hidden_states,
                             compressor_states,
                             compressor_block_tables,
                             compress_sin,
                             compress_cos,
                             attn_metadata.actual_seq_lengths_query);
    LOG(INFO) << "[DSV4][RoPE][CompressorOutput] layer=" << layer_id
              << " compressed_kv=" << tensor_shape_string(compressed_kv) << "/"
              << tensor_dtype_device_string(compressed_kv);
    dump_rope_tensor(
        rope_layer_dump_dir, "compressor.compressed_kv.output", compressed_kv);
    scatter_by_slot(cmp_kv, cmp_slot, compressed_kv);
  }

  torch::Tensor compress_topk_idxs;
  if (compress_ratio_i == 4 && cmp_kv.defined()) {
    auto index_cache = kv_cache.get_index_cache();
    auto indexer_cache_scale = kv_cache.get_indexer_cache_scale();

    std::tuple<torch::Tensor, torch::Tensor> indexer_states{index_kv_state,
                                                            index_score_state};
    std::tuple<torch::Tensor, torch::Tensor> indexer_block_tables{
        index_kv_block_table, index_score_block_table};
    auto indexer_metadata = build_indexer_attention_metadata(
        attn_metadata, index_block_table, index_slot, isprefill);
    CHECK(qli_metadata.defined()) << "DSAttention requires precomputed "
                                     "qli_metadata for compress_ratio==4.";
    auto qli_metadata_opt = std::optional<torch::Tensor>(qli_metadata);
    LOG(INFO) << "[DSV4][RoPE][IndexerInputs] layer=" << layer_id
              << " cos=" << tensor_shape_string(cos) << "/"
              << tensor_dtype_device_string(cos)
              << " sin=" << tensor_shape_string(sin) << "/"
              << tensor_dtype_device_string(sin)
              << " c4_cos=" << tensor_shape_string(attn_metadata.c4_cos) << "/"
              << tensor_dtype_device_string(attn_metadata.c4_cos)
              << " c4_sin=" << tensor_shape_string(attn_metadata.c4_sin) << "/"
              << tensor_dtype_device_string(attn_metadata.c4_sin);
    dump_rope_tensor(rope_layer_dump_dir, "indexer.qr.input", qr);
    dump_rope_tensor(rope_layer_dump_dir, "indexer.cos.input", cos);
    dump_rope_tensor(rope_layer_dump_dir, "indexer.sin.input", sin);
    dump_rope_tensor(
        rope_layer_dump_dir, "indexer.c4_cos.input", attn_metadata.c4_cos);
    dump_rope_tensor(
        rope_layer_dump_dir, "indexer.c4_sin.input", attn_metadata.c4_sin);
    compress_topk_idxs =
        indexer_->select_qli(hidden_states,
                             qr,
                             index_cache,
                             &indexer_cache_scale,
                             indexer_metadata,
                             cos,
                             sin,
                             attn_metadata.c4_cos,
                             attn_metadata.c4_sin,
                             attn_metadata.actual_seq_lengths_query,
                             attn_metadata.actual_seq_lengths_kv,
                             qli_metadata_opt,
                             isprefill,
                             &indexer_states,
                             &indexer_block_tables);
    LOG(INFO) << "[DSV4][RoPE][IndexerOutput] layer=" << layer_id
              << " compress_topk_idxs="
              << tensor_shape_string(compress_topk_idxs) << "/"
              << tensor_dtype_device_string(compress_topk_idxs);
    dump_rope_tensor(rope_layer_dump_dir,
                     "indexer.compress_topk_idxs.output",
                     compress_topk_idxs);
    CHECK(compress_topk_idxs.defined())
        << "DSAttention indexer returned undefined topk indices for "
           "compress_ratio==4.";
  }

  // 7) sparse shared-kv attention
  c10::optional<torch::Tensor> sparse_metadata = c10::nullopt;
  if (compress_ratio_i == 1) {
    sparse_metadata = as_optional(c1_metadata);
  } else if (compress_ratio_i == 4) {
    sparse_metadata = as_optional(c4_metadata);
  } else if (compress_ratio_i == 128) {
    sparse_metadata = as_optional(c128_metadata);
  }

  CHECK(sparse_metadata.has_value())
      << "DSAttention requires precomputed sparse metadata for compress_ratio="
      << compress_ratio_i;
  LOG(INFO)
      << "[DSV4][SparseAttn][Inputs] layer=" << layer_id
      << " q=" << tensor_shape_string(q) << "/" << tensor_dtype_device_string(q)
      << " ori_kv=" << tensor_shape_string(ori_kv) << "/"
      << tensor_dtype_device_string(ori_kv)
      << " cmp_kv=" << tensor_shape_string(cmp_kv) << "/"
      << tensor_dtype_device_string(cmp_kv)
      << " compress_topk_idxs=" << tensor_shape_string(compress_topk_idxs)
      << "/" << tensor_dtype_device_string(compress_topk_idxs)
      << " actual_seq_lengths_query="
      << tensor_shape_string(attn_metadata.actual_seq_lengths_query) << "/"
      << tensor_dtype_device_string(attn_metadata.actual_seq_lengths_query)
      << " actual_seq_lengths_kv="
      << tensor_shape_string(attn_metadata.actual_seq_lengths_kv) << "/"
      << tensor_dtype_device_string(attn_metadata.actual_seq_lengths_kv)
      << " sparse_metadata=" << tensor_shape_string(sparse_metadata.value())
      << "/" << tensor_dtype_device_string(sparse_metadata.value());
  dump_module_tensor("sparse_attn_inputs", "q", q);
  dump_module_tensor("sparse_attn_inputs", "ori_kv", ori_kv);
  dump_module_tensor("sparse_attn_inputs", "cmp_kv", cmp_kv);
  dump_module_tensor(
      "sparse_attn_inputs", "compress_topk_idxs", compress_topk_idxs);
  dump_module_tensor("sparse_attn_inputs",
                     "actual_seq_lengths_query",
                     attn_metadata.actual_seq_lengths_query);
  dump_module_tensor("sparse_attn_inputs",
                     "actual_seq_lengths_kv",
                     attn_metadata.actual_seq_lengths_kv);
  dump_module_tensor(
      "sparse_attn_inputs", "sparse_metadata", sparse_metadata.value());

  auto [attn_output, output_lse] = xllm::kernel::npu::sparse_attn_sharedkv(
      /*q=*/q,
      /*ori_kv=*/as_optional(ori_kv),
      /*cmp_kv=*/compress_ratio_i > 1 ? as_optional(cmp_kv) : c10::nullopt,
      /*ori_sparse_indices=*/c10::nullopt,
      /*cmp_sparse_indices=*/
      compress_ratio_i == 4 ? as_optional(compress_topk_idxs) : c10::nullopt,
      /*ori_block_table=*/as_optional(ori_block_table),
      /*cmp_block_table=*/
      compress_ratio_i > 1 ? as_optional(cmp_block_table) : c10::nullopt,
      /*cu_seqlens_q=*/as_optional(attn_metadata.actual_seq_lengths_query),
      // DeepSeek V4 aligns with MindIE here: runtime uses query cu_seqlens and
      // per-seq KV lengths, not ori/cmp KV cu_seqlens.
      /*cu_seqlens_ori_kv=*/c10::nullopt,
      /*cu_seqlens_cmp_kv=*/c10::nullopt,
      /*seqused_q=*/c10::nullopt,
      /*seqused_kv=*/as_optional(attn_metadata.actual_seq_lengths_kv),
      /*sinks=*/attn_sink_loaded_ ? as_optional(attn_sink_) : c10::nullopt,
      /*metadata=*/sparse_metadata,
      /*softmax_scale=*/softmax_scale_,
      /*cmp_ratio=*/compress_ratio_i,
      /*ori_mask_mode=*/4,
      /*cmp_mask_mode=*/3,
      /*ori_win_left=*/std::max<int64_t>(window_size_ - 1, 0),
      /*ori_win_right=*/0,
      /*layout_q=*/"TND",
      /*layout_kv=*/"PA_ND",
      /*return_softmax_lse=*/false);
  log_node_tensor("sparse_attn.output", attn_output);
  dump_node_tensor("sparse_attn.output", attn_output);
  log_node_tensor("sparse_attn.lse", output_lse);
  dump_node_tensor("sparse_attn.lse", output_lse);

  // 8) output RoPE + projection
  auto o = attn_output.view({-1, n_local_heads_, head_dim_});
  log_node_tensor("o_reshape.input", o);
  dump_node_tensor("o_reshape.input", o);
  apply_partial_rope(o,
                     nope_head_dim_,
                     rope_head_dim_,
                     cos,
                     sin,
                     /*inverse=*/true,
                     layer_id,
                     "o_inverse_rope",
                     rope_layer_dump_dir);
  log_node_tensor("o_after_inverse_rope.output", o);
  dump_node_tensor("o_after_inverse_rope.output", o);

  const int64_t num_tokens = o.size(0);
  auto o_group = o.view({num_tokens, n_local_groups_, -1});
  log_node_tensor("o_group.output", o_group);
  dump_node_tensor("o_group.output", o_group);
  auto wo_a = o_a_proj_->weight().view({n_local_groups_, o_lora_rank_, -1});
  auto o_low_rank = torch::einsum("tgd,grd->tgr", {o_group, wo_a});
  log_node_tensor("o_low_rank.output", o_low_rank);
  dump_node_tensor("o_low_rank.output", o_low_rank);
  auto output = o_b_proj_->forward(o_low_rank.reshape({num_tokens, -1}));
  log_node_tensor("o_b_proj.output", output);
  dump_node_tensor("o_b_proj.output", output);
  std::optional<torch::Tensor> final_lse = std::nullopt;
  (void)output_lse;

  return std::make_tuple(output, final_lse);
}

void DSAttentionImpl::load_state_dict(const StateDict& state_dict) {
  q_a_proj_->load_state_dict(state_dict.get_dict_with_prefix("wq_a."));
  q_b_proj_->load_state_dict(state_dict.get_dict_with_prefix("wq_b."));
  q_layernorm_->load_state_dict(state_dict.get_dict_with_prefix("q_norm."));

  kv_proj_->load_state_dict(state_dict.get_dict_with_prefix("wkv."));
  kv_layernorm_->load_state_dict(state_dict.get_dict_with_prefix("kv_norm."));
  o_a_proj_->load_state_dict(state_dict.get_dict_with_prefix("wo_a."));
  o_b_proj_->load_state_dict(state_dict.get_dict_with_prefix("wo_b."));

  auto attn_sink = state_dict.get_tensor("attn_sink");
  if (!attn_sink.defined()) {
    attn_sink = state_dict.get_tensor("attn_sink.weight");
  }
  if (attn_sink.defined()) {
    if (attn_sink.dim() == 1 && attn_sink.size(0) == num_heads_ &&
        tp_size_ > 1) {
      CHECK_EQ(num_heads_ % tp_size_, 0)
          << "attn_sink full-head tensor size is not divisible by tp_size.";
      const int64_t shard_size = num_heads_ / tp_size_;
      const int64_t shard_start = tp_rank_ * shard_size;
      attn_sink = attn_sink.slice(/*dim=*/0,
                                  /*start=*/shard_start,
                                  /*end=*/shard_start + shard_size);
    }

    CHECK(attn_sink.dim() == 1 && attn_sink.size(0) == n_local_heads_)
        << "attn_sink shape mismatch, expected [" << n_local_heads_ << "], got "
        << attn_sink.sizes();

    torch::NoGradGuard no_grad;
    attn_sink_.copy_(attn_sink.to(attn_sink_.device()).to(attn_sink_.dtype()));
    attn_sink_loaded_ = true;
  }

  if (compressor_ && compress_ratio_ >= 4) {
    auto compressor_state = state_dict.get_dict_with_prefix("compressor.");
    if (compressor_state.size() == 0) {
      compressor_state = state_dict.get_dict_with_prefix("compress.");
    }
    if (compressor_state.size() > 0) {
      compressor_->load_state_dict(compressor_state);
    }
  }

  if (indexer_ && compress_ratio_ == 4) {
    indexer_->load_state_dict(state_dict.get_dict_with_prefix("indexer."));
  }
}

int64_t DSAttentionImpl::non_registered_weight_bytes() const {
  if (!compressor_) {
    return 0;
  }
  return compressor_->weight_bytes();
}

}  // namespace layer
}  // namespace xllm
