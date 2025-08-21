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

#ifndef THIRD_PARTY_KDNN_KDNN_THREADPOOL_H_
#define THIRD_PARTY_KDNN_KDNN_THREADPOOL_H_

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "gflags/gflags.h"

#define EIGEN_USE_THREADS

#include "kdnn.hpp"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/lib/core/status.h"

DECLARE_int64(kdnn_num_threads);

namespace kdnn {

using KDNN::Threading::ThreadpoolWrapper;
using tensorflow::thread::ThreadPool;

class KDNNThreadPool : public ThreadpoolWrapper {
 public:
  KDNNThreadPool() = default;

  KDNNThreadPool(ThreadPool* thread_pool,
                   int num_threads = -1)
      : thread_pool_(thread_pool), 
      eigen_interface_(thread_pool->AsEigenThreadPool()) {
    set_num_and_max_threads(num_threads);
  }

  virtual int GetNumThreads() const override { return num_threads_; }
  virtual bool GetInParallel() const override {
    return (eigen_interface_->CurrentThreadId() != -1) ? true : false;
  }
  virtual void Parallel(int n, int cost,
                        const std::function<void(int, int)>& fn) override {
    thread_pool_->ParallelFor(n, cost, fn);
  }

  ~KDNNThreadPool() {}

 private:
  ThreadPool* thread_pool_ = nullptr;
  Eigen::ThreadPoolInterface* eigen_interface_ = nullptr;
  int num_threads_ = 1;
  inline void set_num_and_max_threads(int num_threads) {
    int pool_threads = eigen_interface_->NumThreads();
    
    if (num_threads > 0) {
      num_threads_ = std::min(pool_threads, num_threads);
      return;
    }
    
    if (FLAGS_kdnn_num_threads > 0) {
      num_threads_ = std::min(pool_threads, static_cast<int>(FLAGS_kdnn_num_threads));
      return;
    }
    
    num_threads_ = pool_threads;
  }
};

}  // namespace kdnn

#endif  // THIRD_PARTY_KDNN_KDNN_THREADPOOL_H_
