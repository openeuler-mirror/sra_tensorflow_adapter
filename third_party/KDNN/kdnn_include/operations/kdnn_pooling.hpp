/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Pooling op
 * Author: KPL
 * Create: 2024-07-19
 * Notes: NA
 */

#ifndef KDNN_POOLING_HPP
#define KDNN_POOLING_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"
#include "types/kdnn_propagation_type.hpp"

namespace KDNN {

enum class PoolingFunction {
    UNDEFINED,
    MAX,
    AVG_EXCLUDE_PADDING,
    AVG_INCLUDE_PADDING,
};

namespace Detail {

class PoolingLayerFWDImpl;
class PoolingLayerBWDImpl;

} // Detail

class KDNN_API_PUBLIC PoolingLayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const Propagation &propKind, const PoolingFunction &poolingAlg,
        const TensorInfo &srcInfo, const TensorInfo &dstInfo, const Shape &strides,
        const Shape &kernel, const Shape &dilation, const Shape &paddingL, const Shape &paddingR) noexcept;
    PoolingLayerFWD(const Propagation &propKind, const PoolingFunction &poolingAlg,
        const TensorInfo &srcInfo, const TensorInfo &dstInfo, const Shape &strides,
        const Shape &kernel, const Shape &dilation, const Shape &paddingL, const Shape &paddingR) noexcept(false);
    PoolingLayerFWD(const PoolingLayerFWD &other) noexcept(false);
    PoolingLayerFWD(PoolingLayerFWD &&other) noexcept;
    PoolingLayerFWD& operator=(const PoolingLayerFWD &other) noexcept(false);
    PoolingLayerFWD& operator=(PoolingLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst, void *ws = nullptr) const noexcept(false);
    void SetPadType(Element::Type newPadType) noexcept;
    ~PoolingLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::PoolingLayerFWDImpl> pImpl;
};

class KDNN_API_PUBLIC PoolingLayerBWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const PoolingFunction &poolingAlg,
        const TensorInfo &srcDiffInfo, const TensorInfo &dstDiffInfo, const Shape &strides,
        const Shape &kernel, const Shape &dilation, const Shape &paddingL, const Shape &paddingR) noexcept;
    PoolingLayerBWD(const PoolingFunction &poolingAlg,
        const TensorInfo &srcInfo, const TensorInfo &dstInfo, const Shape &strides,
        const Shape &kernel, const Shape &dilation, const Shape &paddingL, const Shape &paddingR) noexcept(false);
    PoolingLayerBWD(const PoolingLayerBWD &other) noexcept(false);
    PoolingLayerBWD(PoolingLayerBWD &&other) noexcept;
    PoolingLayerBWD& operator=(const PoolingLayerBWD &other) noexcept(false);
    PoolingLayerBWD& operator=(PoolingLayerBWD &&other) noexcept;
    void Run(void *diffSrc, const void *diffDst, const void *ws = nullptr) const noexcept(false);
    ~PoolingLayerBWD() noexcept;
private:
    std::unique_ptr<Detail::PoolingLayerBWDImpl> pImpl;
};

} // KDNN

#endif // KDNN_POOLING_HPP
