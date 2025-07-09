/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Gemm Operation
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_GEMM_HPP
#define KDNN_GEMM_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

namespace Detail {

class GemmImpl;

} // Detail

class KDNN_API_PUBLIC Gemm final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
        const TensorInfo &cInfo, const TensorInfo &biasInfo) noexcept;
    static Status ValidateInput(const TensorInfo &aInfo, const TensorInfo &bInfo,
        const TensorInfo &cInfo) noexcept;
    Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
        const TensorInfo &biasInfo) noexcept(false);
    Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo,
        const TensorInfo &biasInfo, void *thread_pool) noexcept(false);
    Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo, const TensorInfo &cInfo) noexcept(false);
    Gemm(const TensorInfo &aInfo, const TensorInfo &bInfo,
        const TensorInfo &cInfo, void *thread_pool) noexcept(false);
    Gemm(const Gemm &other) noexcept(false);
    Gemm(Gemm &&other) noexcept;
    Gemm& operator=(const Gemm &other) noexcept(false);
    Gemm& operator=(Gemm &&other) noexcept;
    void Run(const void *a, const void *b, void *c, const void *bias,
        float alpha = 1.0f, float beta = 0.0f) const noexcept(false);
    void Run(const void *a, const void *b, void *c,
        float alpha = 1.0f, float beta = 0.0f) const noexcept(false);
    ~Gemm() noexcept;
private:
    std::unique_ptr<Detail::GemmImpl> pImpl;
};

} // KDNN

#endif // KDNN_GEMM_HPP
