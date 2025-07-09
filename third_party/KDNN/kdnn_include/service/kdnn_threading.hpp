/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Error codes
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_THREADING_HPP
#define KDNN_THREADING_HPP

#include <cassert>
#include <functional>

namespace KDNN {
namespace Threading {

struct ThreadpoolWrapper {
    /// Returns the number of worker threads.
    virtual int GetNumThreads() const = 0;

    /// Returns true if the calling thread belongs to this threadpool.
    virtual bool GetInParallel() const = 0;

    virtual void Parallel(int n, const std::function<void(int, int)> &fn) = 0;

    virtual ~ThreadpoolWrapper() {}
};

enum class EnvMode {
    KDNN_THREAD_USE_ENV,
    KDNN_THREAD_IGNORE_ENV
};

enum class ThreadingControl {
    KDNN_DEFAULT,
    KDNN_MANUAL
};

KDNN_API_PUBLIC bool ShouldCalculateOptThreads() noexcept;
KDNN_API_PUBLIC int SetMaxNumThreads(int nThreads) noexcept;
KDNN_API_PUBLIC int SetFixedNumThreads(int nThreads) noexcept;
KDNN_API_PUBLIC int GetMaxNumThreads() noexcept;
KDNN_API_PUBLIC int GetMaxNumThreads(void* thread_pool) noexcept;
KDNN_API_PUBLIC int SetNumThreadsLocal(int nThreads) noexcept;
KDNN_API_PUBLIC EnvMode SetEnvMode(EnvMode mode) noexcept;
KDNN_API_PUBLIC ThreadingControl GetThreadingControlStatus() noexcept;

} // Threading
} // KDNN

#endif // KDNN_THREADING_HPP
