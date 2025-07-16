/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Concat
 * Author: KPL
 * Create: 2024-07-01
 * Notes: NA
 */

#ifndef KDNN_CONCAT_HPP
#define KDNN_CONCAT_HPP

#include <memory>
#include <vector>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

namespace Detail {

class ConcatLayerImpl;

} // Detail

class KDNN_API_PUBLIC ConcatLayer final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const std::vector<TensorInfo> &srcInfo,
        const int concatDim, const TensorInfo &dstInfo) noexcept;
    ConcatLayer(const std::vector<TensorInfo> &srcInfo, const int concatDim,
        const TensorInfo &dstInfo) noexcept(false);
    ConcatLayer(const ConcatLayer &other) noexcept(false);
    ConcatLayer(ConcatLayer &&other) noexcept;
    ConcatLayer& operator=(const ConcatLayer &other) noexcept(false);
    ConcatLayer& operator=(ConcatLayer &&other) noexcept;
    void Run(const void **src, void *dst) const noexcept(false);
    ~ConcatLayer() noexcept;
private:
    std::unique_ptr<Detail::ConcatLayerImpl> pImpl;
};

} // KDNN

#endif // KDNN_CONCAT_HPP
