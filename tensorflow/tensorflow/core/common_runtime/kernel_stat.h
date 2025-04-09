/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_KERNEL_STAT_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_KERNEL_STAT_H_

#include <atomic>
#include <memory>
#include <queue>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/graph_view.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace ExecutorInternal {

// Stores execution time information about the kernels in an executor's graph.
class KernelStats {
  public:
  KernelStats() = default;

  void Initialize(const GraphView& gview) {
    is_expensive_.resize(gview.num_nodes());
    cost_estimates_ =
        std::make_unique<std::atomic_uint_fast64_t[]>(gview.num_nodes());
    for (int32_t i = 0; i < gview.num_nodes(); ++i) {
      if (gview.node(i)) {
        is_expensive_[i] =
            gview.node(i)->kernel && gview.node(i)->kernel->IsExpensive();
        cost_estimates_[i] = kInitialCostEstimateCycles;
      }
    }
  }

  // Returns true iff the given node is considered "expensive". The
  // executor uses this flag to optimize graph execution, for example
  // by "inlining" inexpensive kernels.
  bool IsExpensive(const NodeItem& node) const {
    return is_expensive_[node.node_id] &&
            (cost_estimates_[node.node_id].load(std::memory_order_relaxed) >
            kOpIsExpensiveThresholdCycles);
  }

  // Returns the value of kernel->IsExpensive().
  bool HasExpensiveMarker(const NodeItem& node) const {
    return is_expensive_[node.node_id];
  }

  // Updates the dynamic cost estimate, which is used to determine whether the
  // given node is expensive. The new cost estimate is a weighted average of
  // the old cost estimate and the latest cost. We only update cost estimates
  // for kernels for which IsExpensive() return true.
  void UpdateCostEstimate(const NodeItem& node, uint64 elapsed_cycles) {
    // N.B. Updates to `cost_estimate` are atomic but unlocked.  Simultaneous
    // updates may result in one or more updates being ignored.  This does not
    // affect correctness but may slow down the update frequency.
    std::atomic_uint_fast64_t& cost_estimate = cost_estimates_[node.node_id];
    auto prev_estimate = cost_estimate.load(std::memory_order_relaxed);

    uint64 new_estimate =
        ((kCostDecay - 1) * prev_estimate + elapsed_cycles) / kCostDecay;

    cost_estimate.store(new_estimate, std::memory_order_relaxed);
  }

  private:
  // Initial time (in CPU cycles) we expect an operation to take.  Used to
  // determine whether an operation should be place in a threadpool.
  // Operations start out "expensive".
  static constexpr uint64 kInitialCostEstimateCycles = 100 * 1000 * 1000;
  static constexpr uint64 kOpIsExpensiveThresholdCycles = 8000;
  static constexpr uint64 kCostDecay = 10;

  std::vector<bool> is_expensive_;
  // std::unique_ptr<std::atomic<bool>[]> is_expensive_;
  std::unique_ptr<std::atomic_uint_fast64_t[]> cost_estimates_;
};
} // namespace ExecutorInternal
} // namespace tensorflow

#endif // TENSORFLOW_CORE_COMMON_RUNTIME_KERNEL_STAT_H_