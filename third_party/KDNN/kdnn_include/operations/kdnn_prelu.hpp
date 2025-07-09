/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Prelu op
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_PRELU_HPP
#define KDNN_PRELU_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

namespace Detail {

class PReLULayerFWDImpl;
class PReLULayerBWDImpl;

} // Detail

class KDNN_API_PUBLIC PReLULayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &weightsInfo,
        const TensorInfo &dstInfo) noexcept;
    PReLULayerFWD(const TensorInfo &srcInfo, const TensorInfo &weightsInfo,
        const TensorInfo &dstInfo) noexcept(false);
    PReLULayerFWD(const PReLULayerFWD &other) noexcept(false);
    PReLULayerFWD(PReLULayerFWD &&other) noexcept;
    PReLULayerFWD& operator=(const PReLULayerFWD &other) noexcept(false);
    PReLULayerFWD& operator=(PReLULayerFWD &&other) noexcept;
    void Run(const void *src, const void *wei, void *dst) const noexcept(false);
    ~PReLULayerFWD() noexcept;
private:
    std::unique_ptr<Detail::PReLULayerFWDImpl> pImpl;
};

class KDNN_API_PUBLIC PReLULayerBWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &diffSrcInfo,
        const TensorInfo &weightsInfo, const TensorInfo &diffWeightsInfo,
        const TensorInfo &diffDstInfo) noexcept;
    PReLULayerBWD(const TensorInfo &srcInfo, const TensorInfo &diffSrcInfo,
        const TensorInfo &weightsInfo, const TensorInfo &diffWeightsInfo,
        const TensorInfo &diffDstInfo) noexcept(false);
    PReLULayerBWD(const PReLULayerBWD &other) noexcept(false);
    PReLULayerBWD(PReLULayerBWD &&other) noexcept;
    PReLULayerBWD& operator=(const PReLULayerBWD &other) noexcept(false);
    PReLULayerBWD& operator=(PReLULayerBWD &&other) noexcept;
    void Run(const void *src, void *diffSrc, const void *wei,
        void *diffWeights, const void *diffDst) const noexcept(false);
    ~PReLULayerBWD() noexcept;
private:
    std::unique_ptr<Detail::PReLULayerBWDImpl> pImpl;
};

} // KDNN

#endif // KDNN_PRELU_HPP
