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

#include "base_executor_impl.h"

#include <glog/logging.h>

#include "common/metrics.h"
#include "core/util/tensor_helper.h"

namespace xllm {

BaseExecutorImpl::BaseExecutorImpl(CausalLM* model,
                                   const ModelArgs& args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {}

ForwardInput BaseExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput BaseExecutorImpl::run(const torch::Tensor& tokens,
                                  const torch::Tensor& positions,
                                  std::vector<KVCache>& kv_caches,
                                  const ModelInputParams& params) {
  LOG(INFO) << "[BaseExecutor::eager] input_params:"
            << " num_sequences=" << params.num_sequences
            << " actual_num_sequences=" << params.actual_num_sequences
            << " kv_max_seq_len=" << params.kv_max_seq_len
            << " q_max_seq_len=" << params.q_max_seq_len
            << " enable_graph=" << params.enable_graph
            << " batch_forward_type=" << params.batch_forward_type.to_string()
            << " tokens_size=" << tokens.size(0)
            << " positions_size=" << positions.size(0);
  LOG(INFO) << "[BaseExecutor::eager] kv_seq_lens_vec="
            << params.kv_seq_lens_vec;
  LOG(INFO) << "[BaseExecutor::eager] q_seq_lens_vec="
            << params.q_seq_lens_vec;
  LOG(INFO) << "[BaseExecutor::eager] dp_global_token_nums="
            << params.dp_global_token_nums;
  print_tensor(params.kv_seq_lens, "[BaseExecutor::eager] kv_seq_lens", 10);
  print_tensor(params.q_seq_lens, "[BaseExecutor::eager] q_seq_lens", 10);
  print_tensor(params.new_cache_slots,
               "[BaseExecutor::eager] new_cache_slots", 10);
  print_tensor(params.block_tables, "[BaseExecutor::eager] block_tables", 10);
  print_tensor(params.q_cu_seq_lens, "[BaseExecutor::eager] q_cu_seq_lens", 10);
  if (params.input_embedding.defined()) {
    print_tensor(params.input_embedding,
                 "[BaseExecutor::eager] input_embedding", 10);
    LOG(INFO) << "[BaseExecutor::eager] input_embedding shape: "
              << params.input_embedding.sizes()
              << " dtype: " << params.input_embedding.dtype()
              << " device: " << params.input_embedding.device();
  }
  if (params.attn_metadata) {
    LOG(INFO) << "[BaseExecutor::eager] attn_metadata is set (non-null)";
  } else {
    LOG(INFO) << "[BaseExecutor::eager] attn_metadata is null";
  }
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}

}  // namespace xllm
