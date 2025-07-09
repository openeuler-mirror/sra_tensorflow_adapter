/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Sum
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_SUM_HPP
#define KDNN_SUM_HPP

#include <memory>
#include <vector>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

namespace Detail {

class SumLayerImpl;

} // Detail

class KDNN_API_PUBLIC SumLayer final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const std::vector<TensorInfo> &srcInfo,
        const float *scales, const TensorInfo &dstInfo) noexcept;
    SumLayer(const std::vector<TensorInfo> &srcInfo, const float *scales,
        const TensorInfo &dstInfo) noexcept(false);
    SumLayer(const SumLayer &other) noexcept(false);
    SumLayer(SumLayer &&other) noexcept;
    SumLayer& operator=(const SumLayer &other) noexcept(false);
    SumLayer& operator=(SumLayer &&other) noexcept;
    void Run(const void **src, void *dst) const noexcept(false);
    ~SumLayer() noexcept;
private:
    std::unique_ptr<Detail::SumLayerImpl> pImpl;
};

} // KDNN

#endif // KDNN_SUM_HPP
