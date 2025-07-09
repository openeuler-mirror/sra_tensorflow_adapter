/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: convolution impl
 * Author: KPL
 * Create: 2024-04-27
 * Notes: NA
 */

#ifndef KDNN_CONVOLUTION_HPP
#define KDNN_CONVOLUTION_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

enum class ConvolutionAlgorithm {
    UNIMPLEMENTED,
    AUTO,
    DIRECT,
    WINOGRAD
};

namespace Detail {

class ConvolutionLayerFWDImpl;
class ConvolutionLayerBWDDataImpl;
class ConvolutionLayerBWDWeightsImpl;
class ConvolutionJITFWDImpl;

} // Detail

class KDNN_API_PUBLIC ConvolutionLayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept;
    ConvolutionLayerFWD(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias,
        const Shape &strides, const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept(false);
    ConvolutionLayerFWD(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept(false);
    ConvolutionLayerFWD(const ConvolutionLayerFWD &other) noexcept(false);
    ConvolutionLayerFWD(ConvolutionLayerFWD &&other) noexcept;
    ConvolutionLayerFWD& operator=(const ConvolutionLayerFWD &other) noexcept(false);
    ConvolutionLayerFWD& operator=(ConvolutionLayerFWD &&other) noexcept;
    void Run(const void *src, const void *wei, void *dst, const void *bia) const noexcept(false);
    ~ConvolutionLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::ConvolutionLayerFWDImpl> pImpl;
};

class KDNN_API_PUBLIC ConvolutionLayerBWDData final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept;
    ConvolutionLayerBWDData(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc,
        const Shape &strides, const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept(false);
    ConvolutionLayerBWDData(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept(false);
    ConvolutionLayerBWDData(const ConvolutionLayerBWDData &other) noexcept(false);
    ConvolutionLayerBWDData(ConvolutionLayerBWDData &&other) noexcept;
    ConvolutionLayerBWDData& operator=(const ConvolutionLayerBWDData &other) noexcept(false);
    ConvolutionLayerBWDData& operator=(ConvolutionLayerBWDData &&other) noexcept;
    void Run(const void *diffDst, const void *wei, void *diffSrc) const noexcept(false);
    ~ConvolutionLayerBWDData() noexcept;
private:
    std::unique_ptr<Detail::ConvolutionLayerBWDDataImpl> pImpl;
};

class KDNN_API_PUBLIC ConvolutionLayerBWDWeights final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept;
    ConvolutionLayerBWDWeights(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias,
        const Shape &strides, const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept(false);
    ConvolutionLayerBWDWeights(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias,
        const Shape &strides, const Shape &dilates,
        const Shape &paddingL, const Shape &paddingR,
        ConvolutionAlgorithm alg) noexcept(false);
    ConvolutionLayerBWDWeights(const ConvolutionLayerBWDWeights &other) noexcept(false);
    ConvolutionLayerBWDWeights(ConvolutionLayerBWDWeights &&other) noexcept;
    ConvolutionLayerBWDWeights& operator=(const ConvolutionLayerBWDWeights &other) noexcept(false);
    ConvolutionLayerBWDWeights& operator=(ConvolutionLayerBWDWeights &&other) noexcept;
    void Run(const void *diffDst, const void *src, void *diffWei, void *diffBias) const noexcept(false);
    ~ConvolutionLayerBWDWeights() noexcept;
private:
    std::unique_ptr<Detail::ConvolutionLayerBWDWeightsImpl> pImpl;
};

class KDNN_API_PUBLIC ConvolutionJITFWD final {
public:
    static Status ValidateInput(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias,
        const Shape &strides, const Shape &dilates, const Shape &paddingL, int kCpuIsa) noexcept;
    explicit ConvolutionJITFWD(int isa, const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias, KDNN::Shape strides, KDNN::Shape paddingL,
        KDNN::Shape dilates, KDNN::Propagation prop, int nthreads, bool isDataLayoutNxc) noexcept(false);
    void ExecuteForwardDW(const float *src, const float *wei,
        float *dst, const float *bias) noexcept;
    void ExecuteForward1D(const float *src, const float *wei,
        float *dst, const float *bias) noexcept;
    void ExecuteForward2D(const float *src, const float *wei,
        float *dst, const float *bias) noexcept;
    void ExecuteForward3D(const float *src, const float *wei,
        float *dst, const float *bias) noexcept;
    ConvolutionJITFWD(const ConvolutionJITFWD &other) noexcept(false);
    ConvolutionJITFWD(ConvolutionJITFWD &&other) noexcept;
    ConvolutionJITFWD& operator=(const ConvolutionJITFWD &other) noexcept(false);
    ConvolutionJITFWD& operator=(ConvolutionJITFWD &&other) noexcept;
    bool CreateKernel();
    ~ConvolutionJITFWD() noexcept;
private:
    std::unique_ptr<Detail::ConvolutionJITFWDImpl> pImpl;
};

} // KDNN

#endif // KDNN_CONVOLUTION_HPP
