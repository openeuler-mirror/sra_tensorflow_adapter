/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: shuffle correlation function
 * Author: KPL
 * Create: 2024-07-20
 * Notes: NA
 */

#ifndef KDNN_SHUFFLE_HPP
#define KDNN_SHUFFLE_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

namespace Detail {

class ShuffleLayerImpl;

} // Detail

class KDNN_API_PUBLIC ShuffleLayer final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo, SizeType axis,
        SizeType groupSize) noexcept;
    ShuffleLayer(const TensorInfo &srcInfo, const TensorInfo &dstInfo, SizeType axis,
        SizeType group) noexcept(false);
    ShuffleLayer(const ShuffleLayer &other) noexcept(false);
    ShuffleLayer(ShuffleLayer &&other) noexcept;
    ShuffleLayer& operator=(const ShuffleLayer &other) noexcept(false);
    ShuffleLayer& operator=(ShuffleLayer &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~ShuffleLayer() noexcept;
private:
    std::unique_ptr<Detail::ShuffleLayerImpl> pImpl;
};

} // KDNN

#endif // KDNN_SHUFFLE_HPP
