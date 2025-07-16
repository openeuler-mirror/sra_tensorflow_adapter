/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Softmax
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_SOFTMAX_HPP
#define KDNN_SOFTMAX_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

enum class SoftmaxAlgorithmKind : std::uint32_t {
    SOFTMAX         = 0x0U,
    LOGSOFTMAX      = 0x1U,
};

namespace Detail {

class SoftmaxLayerFWDImpl;
class SoftmaxLayerBWDImpl;

} // Detail

class KDNN_API_PUBLIC SoftmaxLayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo, SizeType axis,
        SoftmaxAlgorithmKind algorithm) noexcept;
    SoftmaxLayerFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo, SizeType axis,
        SoftmaxAlgorithmKind algorithm) noexcept(false);
    SoftmaxLayerFWD(const SoftmaxLayerFWD &other) noexcept(false);
    SoftmaxLayerFWD(SoftmaxLayerFWD &&other) noexcept;
    SoftmaxLayerFWD& operator=(const SoftmaxLayerFWD &other) noexcept(false);
    SoftmaxLayerFWD& operator=(SoftmaxLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~SoftmaxLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::SoftmaxLayerFWDImpl> pImpl;
};

class KDNN_API_PUBLIC SoftmaxLayerBWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &dstInfo, const TensorInfo &dstDiffInfo,
                                const TensorInfo &srcDiffInfo, SizeType axis, SoftmaxAlgorithmKind algorithm) noexcept;
    SoftmaxLayerBWD(const TensorInfo &dstInfo, const TensorInfo &dstDiffInfo, const TensorInfo &srcDiffInfo,
	                SizeType axis, SoftmaxAlgorithmKind algorithm) noexcept(false);
    SoftmaxLayerBWD(const SoftmaxLayerBWD &other) noexcept(false);
    SoftmaxLayerBWD(SoftmaxLayerBWD &&other) noexcept;
    SoftmaxLayerBWD& operator=(const SoftmaxLayerBWD &other) noexcept(false);
    SoftmaxLayerBWD& operator=(SoftmaxLayerBWD &&other) noexcept;
    void Run(const void *dst, const void *dstDiff, void *srcDiff) const noexcept(false);
    ~SoftmaxLayerBWD() noexcept;
private:
    std::unique_ptr<Detail::SoftmaxLayerBWDImpl> pImpl;
};

} // KDNN

#endif // KDNN_SOFTMAX_HPP
