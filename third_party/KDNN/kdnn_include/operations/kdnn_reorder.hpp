/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: reorder
 * Author: KPL
 * Create: 2024-06-25
 * Notes: NA
 */

#ifndef KDNN_REORDER_HPP
#define KDNN_REORDER_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {
namespace Detail {

class ReorderLayerImpl;

} // Detail

class KDNN_API_PUBLIC ReorderLayer final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo) noexcept;
    ReorderLayer(const TensorInfo &srcInfo, const TensorInfo &dstInfo) noexcept(false);
    ReorderLayer(const ReorderLayer &other) noexcept(false);
    ReorderLayer(ReorderLayer &&other) noexcept;
    ReorderLayer& operator=(const ReorderLayer &other) noexcept(false);
    ReorderLayer& operator=(ReorderLayer &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~ReorderLayer() noexcept;
private:
    std::unique_ptr<Detail::ReorderLayerImpl> pImpl;
};

} // KDNN

#endif // KDNN_REORDER_HPP
