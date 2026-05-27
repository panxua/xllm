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

#include <torch/torch.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_input_params.h"
#include "core/util/tensor_helper.h"

namespace xllm {

// DSA cache type enum for DeepSeek V4 multi-cache management
enum class DSACacheType : int32_t {
  TOKEN = 0,           // block allocated by token count / ratio
  SEQUENCE = 1,        // one block per sequence
  SLIDING_WINDOW = 2,  // sliding window, fixed number of blocks per seq
};

// Per-cache metadata within a layer
struct DSACacheInfo {
  int32_t group_id;    // which block manager group this cache belongs to
  DSACacheType type;   // cache type
  int32_t ratio;       // compression ratio
  int32_t block_size;  // block size for this cache
};

// Group-level info
struct DSAGroupInfo {
  DSACacheType type;
  int32_t ratio;
  int32_t block_size;
};

namespace layer {

// DSAMetadata contains DeepSeek V4 sparse attention specific metadata,
// aligned with Python DSAMetadata(AttentionMetadata) class.
// It is built once at the beginning of model forward pass and reused by
// all layers. Use DSAMetadataBuilder to build instances from ModelInputParams.
struct DSAMetadata {
  // ===== Fields from Python AttentionMetadata base class =====
  // seq_lens: kv sequence lengths (context_length)
  torch::Tensor seq_lens;
  // seq_lens_q: query sequence lengths
  torch::Tensor seq_lens_q;
  // attn_mask: attention mask
  torch::Tensor attn_mask;
  // cos_table / sin_table: base RoPE cos/sin tables
  torch::Tensor cos_table;
  torch::Tensor sin_table;

  // ===== DSA-specific fields =====
  // layer_id: current layer (Python per-layer, C++ shared across layers)
  int32_t layer_id = 0;
  // num_speculative_tokens: number of speculative decoding tokens
  int32_t num_speculative_tokens = 0;
  // True when the metadata is consumed by ACL graph forward. Debug paths must
  // not perform host/device copies in this mode.
  bool is_acl_graph = false;

  // cp_input_dict: context-parallel inputs placeholder (reserved, optional)
  std::unordered_map<std::string, torch::Tensor> cp_input_dict;

  // RoPE caches selected for the current layer's q/kv/output RoPE.
  torch::Tensor cos;
  torch::Tensor sin;
  // RoPE caches for compressor/indexer paths, indexed by compressed positions.
  torch::Tensor c4_cos;
  torch::Tensor c4_sin;
  torch::Tensor c128_cos;
  torch::Tensor c128_sin;
  // Main q/kv RoPE tensors for compressed layers at input-token length.
  torch::Tensor c4_input_cos;
  torch::Tensor c4_input_sin;
  torch::Tensor c128_input_cos;
  torch::Tensor c128_input_sin;
  torch::Tensor start_pos;

  // Multi-manager block tables and slot mappings
  // Indexed as [layer_id][cache_idx] after expansion by build_forward_context.
  // Same-group caches share the same underlying tensor (no copy).
  std::vector<std::vector<torch::Tensor>> block_tables;
  std::vector<std::vector<torch::Tensor>> slot_mappings;

  // Host-side max lengths cached alongside the tensors so graph code can
  // avoid scalar reads from device tensors.
  int64_t max_query_len = 0;
  int64_t max_seq_len = 0;

  // Sequence length metadata
  // actual_seq_lengths_kv: (batch_size,) — per-seq kv context length
  torch::Tensor actual_seq_lengths_kv;
  // actual_seq_lengths_query: (batch_size+1,) — cumsum of per-seq query lengths
  //   prefill: pad(cumsum(context_length), (1,0), 0)
  //   decode:  pad(cumsum(ones(batch_size)), (1,0), 0)
  torch::Tensor actual_seq_lengths_query;
  // max_seqlen_kv / max_seqlen_q: max sequence lengths
  torch::Tensor max_seqlen_kv;
  torch::Tensor max_seqlen_q;

  // Compressed positions
  // input_positions: (total_tokens,) — token position IDs
  torch::Tensor input_positions;
  // c4_pad_positions: positions for C4 compressed RoPE
  torch::Tensor c4_pad_positions;
  // c128_pad_positions: positions for C128 compressed RoPE
  torch::Tensor c128_pad_positions;

  // Precomputed sparse/indexer metadata tensors (Python forward aligned).
  // Built once per model forward before layer iteration.
  torch::Tensor c1_metadata;
  torch::Tensor c4_metadata;
  torch::Tensor c128_metadata;
  torch::Tensor qli_metadata;

  // hadamard: Hadamard transform matrix
  torch::Tensor hadamard;

  // Owns the device storage for non-graph DSA metadata tensors packed into a
  // single host-to-device transfer. Individual metadata tensors may be views
  // into this buffer.
  torch::Tensor packed_metadata_buffer;

  // Cache spec per layer
  // caches_info[layer_id][cache_idx] = {group_id, type, ratio, block_size}
  // Points to model-owned data; valid for the lifetime of the model.
  const std::vector<std::vector<DSACacheInfo>>* caches_info = nullptr;
};

}  // namespace layer

// Describes one CPU metadata tensor packed into the shared uint8 buffer and
// records all DSAMetadata fields that should be rebound to its device view.
struct DSAPackedTensorSpec {
  torch::Tensor cpu_tensor;
  std::vector<torch::Tensor*> targets;
  std::vector<int64_t> sizes;
  torch::ScalarType dtype = torch::kUInt8;
  size_t offset = 0;
  size_t nbytes = 0;
};

// Shared graph metadata state for all DSA-family models. DSA MTP must reuse the
// same state and helper functions as ordinary DSA so future metadata fields are
// added once and remain strictly aligned across target and MTP graph paths.
struct DSAGraphMetadataState : ModelGraphMetadataState {
  std::string model_type;
  torch::Tensor packed_metadata_host_buffer;
  torch::Tensor packed_metadata_buffer;
  torch::Tensor attn_mask;
  torch::Tensor start_pos;

  explicit DSAGraphMetadataState(std::string model_type_value = "DSA")
      : model_type(std::move(model_type_value)) {}
};

// Low-level helper
inline torch::Tensor dsa_maybe_to_device(const torch::Tensor& tensor,
                                         const torch::Device& device) {
  if (!tensor.defined() || tensor.device() == device) {
    return tensor;
  }
  return tensor.to(device);
}

inline size_t dsa_align_up(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

inline void dsa_add_packed_tensor(torch::Tensor& tensor,
                                  const torch::Device& runtime_device,
                                  std::vector<DSAPackedTensorSpec>& specs) {
  if (!tensor.defined() || tensor.device() == runtime_device) {
    return;
  }
  if (!tensor.device().is_cpu() || tensor.numel() == 0) {
    tensor = dsa_maybe_to_device(tensor, runtime_device);
    return;
  }

  if (tensor.is_contiguous()) {
    const size_t nbytes =
        static_cast<size_t>(tensor.numel() * tensor.element_size());
    for (DSAPackedTensorSpec& spec : specs) {
      if (spec.cpu_tensor.data_ptr() == tensor.data_ptr() &&
          spec.nbytes == nbytes && spec.dtype == tensor.scalar_type() &&
          spec.sizes == tensor.sizes().vec()) {
        spec.targets.push_back(&tensor);
        return;
      }
    }
  }

  torch::Tensor contiguous = tensor.contiguous();
  const size_t nbytes =
      static_cast<size_t>(contiguous.numel() * contiguous.element_size());
  for (DSAPackedTensorSpec& spec : specs) {
    if (spec.cpu_tensor.data_ptr() == contiguous.data_ptr() &&
        spec.nbytes == nbytes && spec.dtype == contiguous.scalar_type() &&
        spec.sizes == contiguous.sizes().vec()) {
      spec.targets.push_back(&tensor);
      return;
    }
  }

  DSAPackedTensorSpec spec;
  spec.cpu_tensor = std::move(contiguous);
  spec.targets.push_back(&tensor);
  spec.sizes = spec.cpu_tensor.sizes().vec();
  spec.dtype = spec.cpu_tensor.scalar_type();
  spec.nbytes = nbytes;
  specs.push_back(std::move(spec));
}

inline void dsa_collect_cpu_metadata_tensors(
    layer::DSAMetadata& dsa,
    const torch::Device& runtime_device,
    std::vector<DSAPackedTensorSpec>& specs) {
  dsa_add_packed_tensor(dsa.seq_lens, runtime_device, specs);
  dsa_add_packed_tensor(dsa.seq_lens_q, runtime_device, specs);
  dsa_add_packed_tensor(dsa.actual_seq_lengths_query, runtime_device, specs);
  dsa_add_packed_tensor(dsa.actual_seq_lengths_kv, runtime_device, specs);
  dsa_add_packed_tensor(dsa.max_seqlen_q, runtime_device, specs);
  dsa_add_packed_tensor(dsa.max_seqlen_kv, runtime_device, specs);
  dsa_add_packed_tensor(dsa.input_positions, runtime_device, specs);
  dsa_add_packed_tensor(dsa.c4_pad_positions, runtime_device, specs);
  dsa_add_packed_tensor(dsa.c128_pad_positions, runtime_device, specs);

  for (std::vector<torch::Tensor>& layer_block_tables : dsa.block_tables) {
    for (torch::Tensor& block_table : layer_block_tables) {
      dsa_add_packed_tensor(block_table, runtime_device, specs);
    }
  }
  for (std::vector<torch::Tensor>& layer_slot_mappings : dsa.slot_mappings) {
    for (torch::Tensor& slot_mapping : layer_slot_mappings) {
      dsa_add_packed_tensor(slot_mapping, runtime_device, specs);
    }
  }
  dsa_add_packed_tensor(dsa.hadamard, runtime_device, specs);
}

inline size_t dsa_layout_packed_tensor_specs(
    std::vector<DSAPackedTensorSpec>& specs) {
  static constexpr size_t kAlignment = 64;
  size_t total_bytes = 0;
  for (DSAPackedTensorSpec& spec : specs) {
    total_bytes = dsa_align_up(total_bytes, kAlignment);
    spec.offset = total_bytes;
    total_bytes += spec.nbytes;
  }
  return total_bytes;
}

inline torch::Tensor dsa_build_packed_host_buffer(
    const std::vector<DSAPackedTensorSpec>& specs,
    size_t total_bytes) {
  torch::Tensor host_buffer =
      torch::empty({static_cast<int64_t>(total_bytes)},
                   torch::TensorOptions()
                       .dtype(torch::kUInt8)
                       .device(torch::kCPU)
                       .pinned_memory(true));
  uint8_t* host_ptr = static_cast<uint8_t*>(host_buffer.data_ptr());
  for (const DSAPackedTensorSpec& spec : specs) {
    std::memcpy(
        host_ptr + spec.offset, spec.cpu_tensor.data_ptr(), spec.nbytes);
  }
  return host_buffer;
}

inline void dsa_fill_packed_host_buffer(
    const std::vector<DSAPackedTensorSpec>& specs,
    const torch::Tensor& host_buffer) {
  uint8_t* host_ptr = static_cast<uint8_t*>(host_buffer.data_ptr());
  for (const DSAPackedTensorSpec& spec : specs) {
    std::memcpy(
        host_ptr + spec.offset, spec.cpu_tensor.data_ptr(), spec.nbytes);
  }
}

inline void dsa_bind_packed_tensor_views(
    const std::vector<DSAPackedTensorSpec>& specs,
    const torch::Tensor& device_buffer) {
  const uint8_t* device_ptr =
      static_cast<const uint8_t*>(device_buffer.data_ptr());
  for (const DSAPackedTensorSpec& spec : specs) {
    torch::Tensor view =
        get_tensor_from_blob(spec.sizes, spec.dtype, device_ptr + spec.offset);
    for (torch::Tensor* target : spec.targets) {
      *target = view;
    }
  }
}

inline void dsa_move_metadata_to_device(layer::DSAMetadata& dsa,
                                        const torch::Device& runtime_device) {
  dsa.seq_lens = dsa_maybe_to_device(dsa.seq_lens, runtime_device);
  dsa.seq_lens_q = dsa_maybe_to_device(dsa.seq_lens_q, runtime_device);
  dsa.actual_seq_lengths_query =
      dsa_maybe_to_device(dsa.actual_seq_lengths_query, runtime_device);
  dsa.actual_seq_lengths_kv =
      dsa_maybe_to_device(dsa.actual_seq_lengths_kv, runtime_device);
  dsa.max_seqlen_q = dsa_maybe_to_device(dsa.max_seqlen_q, runtime_device);
  dsa.max_seqlen_kv = dsa_maybe_to_device(dsa.max_seqlen_kv, runtime_device);
  dsa.input_positions =
      dsa_maybe_to_device(dsa.input_positions, runtime_device);
  dsa.c4_pad_positions =
      dsa_maybe_to_device(dsa.c4_pad_positions, runtime_device);
  dsa.c128_pad_positions =
      dsa_maybe_to_device(dsa.c128_pad_positions, runtime_device);

  for (std::vector<torch::Tensor>& layer_block_tables : dsa.block_tables) {
    for (torch::Tensor& block_table : layer_block_tables) {
      block_table = dsa_maybe_to_device(block_table, runtime_device);
    }
  }
  for (std::vector<torch::Tensor>& layer_slot_mappings : dsa.slot_mappings) {
    for (torch::Tensor& slot_mapping : layer_slot_mappings) {
      slot_mapping = dsa_maybe_to_device(slot_mapping, runtime_device);
    }
  }

  dsa.hadamard = dsa_maybe_to_device(dsa.hadamard, runtime_device);
}

inline void dsa_pack_metadata_to_device(layer::DSAMetadata& dsa,
                                        const torch::Device& runtime_device) {
#if defined(USE_NPU)
  if (runtime_device.is_cpu() ||
      runtime_device.type() != c10::DeviceType::PrivateUse1) {
    dsa_move_metadata_to_device(dsa, runtime_device);
    return;
  }

  std::vector<DSAPackedTensorSpec> specs;
  dsa_collect_cpu_metadata_tensors(dsa, runtime_device, specs);
  const size_t total_bytes = dsa_layout_packed_tensor_specs(specs);
  if (total_bytes == 0) {
    return;
  }

  torch::Tensor host_buffer = dsa_build_packed_host_buffer(specs, total_bytes);
  dsa.packed_metadata_buffer = safe_to(
      host_buffer,
      torch::TensorOptions().dtype(torch::kUInt8).device(runtime_device),
      false);
  dsa_bind_packed_tensor_views(specs, dsa.packed_metadata_buffer);
#else
  dsa_move_metadata_to_device(dsa, runtime_device);
#endif
}

// High-level helper
inline bool dsa_tensor_aliases_storage(const torch::Tensor& lhs,
                                       const torch::Tensor& rhs) {
  return lhs.defined() && rhs.defined() && lhs.data_ptr() == rhs.data_ptr() &&
         lhs.sizes() == rhs.sizes() && lhs.strides() == rhs.strides();
}

inline torch::Tensor copy_to_dsa_graph_persistent_tensor(
    const torch::Tensor& src,
    torch::Tensor& dst,
    const std::string& model_type) {
  if (!src.defined()) {
    return src;
  }
  if (!dst.defined()) {
    dst = torch::empty_like(src);
  } else {
    CHECK_EQ(dst.scalar_type(), src.scalar_type())
        << model_type << " graph metadata tensor dtype changed";
    CHECK_EQ(dst.device(), src.device())
        << model_type << " graph metadata tensor device changed";
    if (dst.sizes() != src.sizes()) {
      bool can_copy_into_capacity = dst.dim() == src.dim() && src.dim() > 0 &&
                                    src.size(0) <= dst.size(0);
      for (int64_t dim = 1; can_copy_into_capacity && dim < src.dim(); ++dim) {
        can_copy_into_capacity = dst.size(dim) == src.size(dim);
      }
      CHECK(can_copy_into_capacity)
          << model_type << " graph metadata tensor size changed from "
          << dst.sizes() << " to " << src.sizes();
      dst.zero_();
      dst.slice(/*dim=*/0, /*start=*/0, /*end=*/src.size(0))
          .copy_(src, /*non_blocking=*/true);
      return dst;
    }
  }
  if (!dsa_tensor_aliases_storage(src, dst)) {
    dst.copy_(src, /*non_blocking=*/true);
  }
  return dst;
}

inline void copy_to_dsa_graph_packed_metadata_buffer(
    layer::DSAMetadata& dsa,
    DSAGraphMetadataState& state,
    const torch::Device& runtime_device) {
#if defined(USE_NPU)
  if (runtime_device.is_cpu() ||
      runtime_device.type() != c10::DeviceType::PrivateUse1) {
    dsa_move_metadata_to_device(dsa, runtime_device);
    return;
  }

  std::vector<DSAPackedTensorSpec> specs;
  dsa_collect_cpu_metadata_tensors(dsa, runtime_device, specs);
  const size_t total_bytes = dsa_layout_packed_tensor_specs(specs);
  if (total_bytes == 0) {
    return;
  }

  if (!state.packed_metadata_host_buffer.defined() ||
      state.packed_metadata_host_buffer.scalar_type() != torch::kUInt8 ||
      state.packed_metadata_host_buffer.device() != torch::kCPU ||
      state.packed_metadata_host_buffer.numel() <
          static_cast<int64_t>(total_bytes)) {
    state.packed_metadata_host_buffer =
        torch::empty({static_cast<int64_t>(total_bytes)},
                     torch::TensorOptions()
                         .dtype(torch::kUInt8)
                         .device(torch::kCPU)
                         .pinned_memory(true));
  }
  torch::Tensor host_buffer = state.packed_metadata_host_buffer.slice(
      /*dim=*/0, /*start=*/0, /*end=*/static_cast<int64_t>(total_bytes));
  dsa_fill_packed_host_buffer(specs, host_buffer);
  torch::TensorOptions device_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(runtime_device);
  if (!state.packed_metadata_buffer.defined()) {
    state.packed_metadata_buffer =
        torch::empty({static_cast<int64_t>(total_bytes)}, device_options);
  } else {
    CHECK_EQ(state.packed_metadata_host_buffer.scalar_type(), torch::kUInt8)
        << state.model_type << " graph host packed metadata dtype changed";
    CHECK_EQ(state.packed_metadata_host_buffer.device(), torch::kCPU)
        << state.model_type << " graph host packed metadata device changed";
    CHECK_GE(state.packed_metadata_host_buffer.numel(),
             static_cast<int64_t>(total_bytes))
        << state.model_type
        << " graph host packed metadata exceeds persistent capacity: required="
        << total_bytes
        << ", capacity=" << state.packed_metadata_host_buffer.numel();
    CHECK_EQ(state.packed_metadata_buffer.scalar_type(), torch::kUInt8)
        << state.model_type << " graph packed metadata dtype changed";
    CHECK_EQ(state.packed_metadata_buffer.device(), runtime_device)
        << state.model_type << " graph packed metadata device changed";
    CHECK_GE(state.packed_metadata_buffer.numel(),
             static_cast<int64_t>(total_bytes))
        << state.model_type
        << " graph packed metadata exceeds persistent capacity: required="
        << total_bytes << ", capacity=" << state.packed_metadata_buffer.numel();
  }

  state.packed_metadata_buffer
      .slice(/*dim=*/0,
             /*start=*/0,
             /*end=*/static_cast<int64_t>(total_bytes))
      .copy_(host_buffer, /*non_blocking=*/true);
  dsa.packed_metadata_buffer = state.packed_metadata_buffer.slice(
      /*dim=*/0, /*start=*/0, /*end=*/static_cast<int64_t>(total_bytes));
  dsa_bind_packed_tensor_views(specs, dsa.packed_metadata_buffer);
#else
  dsa_move_metadata_to_device(dsa, runtime_device);
#endif
}

// Normalize DSA graph metadata rows for both ordinary DSA decode and DSA MTP
// validate. Ordinary DSA is the decode_tokens=1 special case, while MTP
// validate uses one row for the target token plus one row per speculative token.
// Rows in [actual_num_tokens, padded_num_tokens) are graph bucket padding and
// are zeroed so DSAMetadataBuilder treats them as inactive.
inline void normalize_dsa_graph_metadata_input_params(
    ModelInputParams& params,
    int64_t decode_tokens) {
  const int64_t tokens_per_sequence = std::max<int64_t>(decode_tokens, 1);
  int64_t actual_num_tokens = params.meta.actual_num_sequences;
  if (actual_num_tokens <= 0 && params.attention.device.q_seq_lens.defined() &&
      params.attention.device.q_seq_lens.dim() >= 1) {
    actual_num_tokens = params.attention.device.q_seq_lens.size(0);
  }
  if (actual_num_tokens <= 0 && params.attention.device.kv_seq_lens.defined() &&
      params.attention.device.kv_seq_lens.dim() >= 1) {
    actual_num_tokens = params.attention.device.kv_seq_lens.size(0);
  }
  if (actual_num_tokens <= 0 && !params.attention.host.q_seq_lens.empty()) {
    actual_num_tokens =
        static_cast<int64_t>(params.attention.host.q_seq_lens.size());
  }
  if (actual_num_tokens <= 0 && !params.attention.host.kv_seq_lens.empty()) {
    actual_num_tokens =
        static_cast<int64_t>(params.attention.host.kv_seq_lens.size());
  }
  if (actual_num_tokens <= 0 && params.attention.device.block_tables.defined() &&
      params.attention.device.block_tables.dim() >= 2) {
    actual_num_tokens = params.attention.device.block_tables.size(0);
  }
  for (const torch::Tensor& block_table : params.multi_block_tables) {
    if (actual_num_tokens > 0) {
      break;
    }
    if (block_table.defined() && block_table.dim() >= 2) {
      actual_num_tokens = block_table.size(0);
    }
  }

  actual_num_tokens = std::max<int64_t>(actual_num_tokens, 0);
  if (tokens_per_sequence > 1 && actual_num_tokens > 0) {
    actual_num_tokens =
        (actual_num_tokens / tokens_per_sequence) * tokens_per_sequence;
  }

  int64_t padded_num_tokens = actual_num_tokens;
  if (params.enable_cuda_graph) {
    padded_num_tokens =
        std::max<int64_t>(padded_num_tokens, params.meta.num_sequences);
  }
  if (padded_num_tokens <= 0) {
    padded_num_tokens = tokens_per_sequence;
  }
  actual_num_tokens = std::min<int64_t>(actual_num_tokens, padded_num_tokens);

  auto trim_lens_vec = [padded_num_tokens,
                        actual_num_tokens](std::vector<int32_t>& lens) {
    if (lens.empty()) {
      lens.assign(static_cast<size_t>(padded_num_tokens), 0);
    } else if (static_cast<int64_t>(lens.size()) < padded_num_tokens) {
      lens.resize(static_cast<size_t>(padded_num_tokens), 0);
    } else {
      lens.resize(static_cast<size_t>(padded_num_tokens));
    }
    std::fill(lens.begin() + actual_num_tokens, lens.end(), 0);
  };

  trim_lens_vec(params.attention.host.kv_seq_lens);
  trim_lens_vec(params.attention.host.q_seq_lens);
  params.meta.num_sequences = static_cast<int32_t>(padded_num_tokens);
  params.meta.actual_num_sequences = static_cast<int32_t>(actual_num_tokens);
}

inline void persist_dsa_graph_metadata(DSAGraphMetadataState& state,
                                       layer::DSAMetadata& dsa) {
  dsa.attn_mask = copy_to_dsa_graph_persistent_tensor(
      dsa.attn_mask, state.attn_mask, state.model_type);
  dsa.start_pos = copy_to_dsa_graph_persistent_tensor(
      dsa.start_pos, state.start_pos, state.model_type);
}

}  // namespace xllm
