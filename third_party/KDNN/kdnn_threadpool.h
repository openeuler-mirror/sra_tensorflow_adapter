#ifndef TENSORFLOW_CORE_UTIL_KDNN_THREADPOOL_H_
#define TENSORFLOW_CORE_UTIL_KDNN_THREADPOOL_H_

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define EIGEN_USE_THREADS

#include "kdnn.hpp"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/env_var.h"

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
    
    tensorflow::int64 env_threads = -1;
    tensorflow::Status status = tensorflow::ReadInt64FromEnvVar("KDNN_NUM_THREADS", -1, &env_threads);
    if (!status.ok()) {
      LOG(WARNING) << "Parse env KDNN_NUM_THREADS failed, use default thread nums";
    }
    if (env_threads > 0) {
      num_threads_ = std::min(pool_threads, static_cast<int>(env_threads));
      return;
    }
    
    num_threads_ = pool_threads;
  }
};

}  // namespace kdnn

#endif  // TENSORFLOW_CORE_UTIL_KDNN_THREADPOOL_H_
