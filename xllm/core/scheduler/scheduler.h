/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <absl/time/time.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "framework/batch/batch.h"
#include "framework/request/request.h"

namespace xllm {

class SchedulerBase {
 public:
  virtual ~SchedulerBase() = default;

  virtual void run() {}

  virtual void stop() {}

  // scheduler forward execute
  virtual void step(const absl::Duration& timeout) = 0;

  // offline running
  virtual void generate() = 0;

  // incr/decr pending requests
  virtual void incr_pending_requests(size_t count) {}
  virtual void decr_pending_requests() {}
  virtual void process_batch_output() {}
  virtual size_t num_pending_requests() { return 0; }
};

class Scheduler : public SchedulerBase {
 public:
  virtual ~Scheduler() = default;

  // add a new request to scheduler.
  virtual bool add_request(std::shared_ptr<Request>& request) = 0;

  virtual uint32_t get_waiting_requests_num() const = 0;

  virtual void get_latency_metrics(std::vector<int64_t>& ttft,
                                   std::vector<int64_t>& tbt) = 0;

  virtual const InstanceInfo& get_instance_info() = 0;
};

// Engine-driven scheduler split interface:
// Stage1 schedule -> Stage2 engine forward -> Stage3 postprocess.
class EngineDrivenScheduler : public Scheduler {
 public:
  struct RuntimeOps {
    std::function<void(std::vector<Batch>&)> run_forward_batch;
    std::function<void(std::vector<Batch>&)> update_last_forward_result;
    std::function<std::vector<int64_t>()> get_active_activation_memory;
  };

  virtual std::vector<Batch> schedule(const absl::Duration& timeout) {
    (void)timeout;
    return {};
  }

  virtual void process_batch_output(std::vector<Batch>& batch) { (void)batch; }

  virtual size_t request_queue_size() const { return 0; }

  virtual void wait_response_completion() {}

  virtual int32_t default_step_timeout_ms() const { return 500; }

  void set_runtime_ops(RuntimeOps runtime_ops) {
    runtime_ops_ = std::move(runtime_ops);
  }

 protected:
  void run_forward_batch(std::vector<Batch>& batch) {
    if (runtime_ops_.run_forward_batch) {
      runtime_ops_.run_forward_batch(batch);
    }
  }

  void update_last_forward_result(std::vector<Batch>& batch) {
    if (runtime_ops_.update_last_forward_result) {
      runtime_ops_.update_last_forward_result(batch);
    }
  }

  std::vector<int64_t> get_active_activation_memory_from_runtime() {
    if (runtime_ops_.get_active_activation_memory) {
      return runtime_ops_.get_active_activation_memory();
    }
    return {};
  }

 private:
  RuntimeOps runtime_ops_;
};

}  // namespace xllm
