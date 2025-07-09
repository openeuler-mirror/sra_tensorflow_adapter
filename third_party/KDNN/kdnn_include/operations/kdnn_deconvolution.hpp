/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: deconvolution impl
 * Author: KPL
 * Create: 2024-05-24
 * Notes: NA
 */

#ifndef KDNN_DECONVOLUTION_HPP
#define KDNN_DECONVOLUTION_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

enum class DeconvolutionAlgorithm {
    UNIMPLEMENTED,
    DIRECT,
    WINOGRAD
};

namespace Detail {

class DeconvolutionLayerFWDImpl;
class DeconvolutionLayerBWDDataImpl;
class DeconvolutionLayerBWDWeightsImpl;

} // Detail

class KDNN_API_PUBLIC DeconvolutionLayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        DeconvolutionAlgorithm alg) noexcept;
    DeconvolutionLayerFWD(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias,
        const Shape &strides, const Shape &paddingL, const Shape &paddingR,
        DeconvolutionAlgorithm alg) noexcept(false);
    DeconvolutionLayerFWD(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        DeconvolutionAlgorithm alg) noexcept(false);
    DeconvolutionLayerFWD(const DeconvolutionLayerFWD &other) noexcept(false);
    DeconvolutionLayerFWD(DeconvolutionLayerFWD &&other) noexcept;
    DeconvolutionLayerFWD& operator=(const DeconvolutionLayerFWD &other) noexcept(false);
    DeconvolutionLayerFWD& operator=(DeconvolutionLayerFWD &&other) noexcept;
    void Run(const void *src, const void *wei, void *dst, const void *bia) const noexcept(false);
    ~DeconvolutionLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::DeconvolutionLayerFWDImpl> pImpl;
};

class KDNN_API_PUBLIC DeconvolutionLayerBWDData final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc, const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR, DeconvolutionAlgorithm alg) noexcept;
    DeconvolutionLayerBWDData(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc, const Shape &strides, const Shape &paddingL,
        const Shape &paddingR, DeconvolutionAlgorithm alg) noexcept(false);
    DeconvolutionLayerBWDData(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc, const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR, DeconvolutionAlgorithm alg) noexcept(false);
    DeconvolutionLayerBWDData(const DeconvolutionLayerBWDData &other) noexcept(false);
    DeconvolutionLayerBWDData(DeconvolutionLayerBWDData &&other) noexcept;
    DeconvolutionLayerBWDData& operator=(const DeconvolutionLayerBWDData &other) noexcept(false);
    DeconvolutionLayerBWDData& operator=(DeconvolutionLayerBWDData &&other) noexcept;
    void Run(const void *diffDst, const void *wei, void *diffSrc) const noexcept(false);
    ~DeconvolutionLayerBWDData() noexcept;
private:
    std::unique_ptr<Detail::DeconvolutionLayerBWDDataImpl> pImpl;
};

class KDNN_API_PUBLIC DeconvolutionLayerBWDWeights final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        DeconvolutionAlgorithm alg) noexcept;
    DeconvolutionLayerBWDWeights(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias,
        const Shape &strides, const Shape &paddingL, const Shape &paddingR,
        DeconvolutionAlgorithm alg) noexcept(false);
    DeconvolutionLayerBWDWeights(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        DeconvolutionAlgorithm alg) noexcept(false);
    DeconvolutionLayerBWDWeights(const DeconvolutionLayerBWDWeights &other) noexcept(false);
    DeconvolutionLayerBWDWeights(DeconvolutionLayerBWDWeights &&other) noexcept;
    DeconvolutionLayerBWDWeights& operator=(const DeconvolutionLayerBWDWeights &other) noexcept(false);
    DeconvolutionLayerBWDWeights& operator=(DeconvolutionLayerBWDWeights &&other) noexcept;
    void Run(const void *diffDst, const void *src, void *diffWei, void *diffBias) const noexcept(false);
    ~DeconvolutionLayerBWDWeights() noexcept;
private:
    std::unique_ptr<Detail::DeconvolutionLayerBWDWeightsImpl> pImpl;
};

} // KDNN

#endif // KDNN_DECONVOLUTION_HPP
