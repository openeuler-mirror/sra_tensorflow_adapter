/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Inner product
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_INNER_PRODUCT_HPP
#define KDNN_INNER_PRODUCT_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

namespace Detail {

class InnerProductLayerFWDImpl;
class InnerProductLayerBWDDataImpl;
class InnerProductLayerBWDWeightsImpl;

} // Detail

class KDNN_API_PUBLIC InnerProductLayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias) noexcept;
    InnerProductLayerFWD(const TensorInfo &src, const TensorInfo &weights,
        const TensorInfo &dst, const TensorInfo &bias) noexcept(false);
    InnerProductLayerFWD(const InnerProductLayerFWD &other) noexcept(false);
    InnerProductLayerFWD(InnerProductLayerFWD &&other) noexcept;
    InnerProductLayerFWD& operator=(const InnerProductLayerFWD &other) noexcept(false);
    InnerProductLayerFWD& operator=(InnerProductLayerFWD &&other) noexcept;
    void Run(const void *src, const void *wei, void *dst, const void *bia = nullptr) const noexcept(false);
    ~InnerProductLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::InnerProductLayerFWDImpl> pImpl;
};

class KDNN_API_PUBLIC InnerProductLayerBWDData final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc) noexcept;
    InnerProductLayerBWDData(const TensorInfo &diffDst, const TensorInfo &weights,
        const TensorInfo &diffSrc) noexcept(false);
    InnerProductLayerBWDData(const InnerProductLayerBWDData &other) noexcept(false);
    InnerProductLayerBWDData(InnerProductLayerBWDData &&other) noexcept;
    InnerProductLayerBWDData& operator=(const InnerProductLayerBWDData &other) noexcept(false);
    InnerProductLayerBWDData& operator=(InnerProductLayerBWDData &&other) noexcept;
    void Run(const void *diffDst, const void *wei, void *diffSrc) const noexcept(false);
    ~InnerProductLayerBWDData() noexcept;
private:
    std::unique_ptr<Detail::InnerProductLayerBWDDataImpl> pImpl;
};

class KDNN_API_PUBLIC InnerProductLayerBWDWeights final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias) noexcept;
    InnerProductLayerBWDWeights(const TensorInfo &diffDst, const TensorInfo &src,
        const TensorInfo &diffWeights, const TensorInfo &diffBias) noexcept(false);
    InnerProductLayerBWDWeights(const InnerProductLayerBWDWeights &other) noexcept(false);
    InnerProductLayerBWDWeights(InnerProductLayerBWDWeights &&other) noexcept;
    InnerProductLayerBWDWeights& operator=(const InnerProductLayerBWDWeights &other) noexcept(false);
    InnerProductLayerBWDWeights& operator=(InnerProductLayerBWDWeights &&other) noexcept;
    void Run(const void *diffDst, const void *src, void *diffWeights, void *diffBias = nullptr) const noexcept(false);
    ~InnerProductLayerBWDWeights() noexcept;
private:
    std::unique_ptr<Detail::InnerProductLayerBWDWeightsImpl> pImpl;
};

} // KDNN

#endif // KDNN_INNER_PRODUCT_HPP
