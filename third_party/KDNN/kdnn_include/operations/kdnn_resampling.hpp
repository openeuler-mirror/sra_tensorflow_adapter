/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: mathematical correlation function
 * Author: KPL
 * Create: 2024-07-3
 * Notes: NA
 */

#ifndef KDNN_RESAMPLING_HPP
#define KDNN_RESAMPLING_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

enum class ResamplingAlg {
    UNIMPLEMENTED,
    NEAREST_NEIGHBOR,
    LINEAR,
};

namespace Detail {
namespace Resampling {
    class ResamplingLayerImplFWD;
    class ResamplingLayerImplBWD;
} // Resampling
} // Detail

class KDNN_API_PUBLIC ResamplingLayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
        ResamplingAlg algKind) noexcept;
    ResamplingLayerFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
        ResamplingAlg algKind) noexcept(false);
    ResamplingLayerFWD(const ResamplingLayerFWD &other) noexcept(false);
    ResamplingLayerFWD(ResamplingLayerFWD &&other) noexcept;
    ResamplingLayerFWD& operator=(const ResamplingLayerFWD &other) noexcept(false);
    ResamplingLayerFWD& operator=(ResamplingLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~ResamplingLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::Resampling::ResamplingLayerImplFWD> pImpl;
};

class KDNN_API_PUBLIC ResamplingLayerBWD final {
public:
    using SizeType = ::KDNN::SizeType;
    ResamplingLayerBWD(const TensorInfo &diffDstInfo, const TensorInfo &diffSrcInfo,
        ResamplingAlg algKind) noexcept(false);

    ResamplingLayerBWD(const ResamplingLayerBWD &other) noexcept(false);
    ResamplingLayerBWD(ResamplingLayerBWD &&other) noexcept;
    ResamplingLayerBWD& operator=(const ResamplingLayerBWD &other) noexcept(false);
    ResamplingLayerBWD& operator=(ResamplingLayerBWD &&other) noexcept;

    ~ResamplingLayerBWD() noexcept;

    static Status ValidateInput(const TensorInfo &diffDstInfo, const TensorInfo &diffSrcInfo,
        ResamplingAlg algKind) noexcept;

    void Run(const void *diffDst, void *diffSrc) const noexcept(false);

private:
    std::unique_ptr<Detail::Resampling::ResamplingLayerImplBWD> pImpl;
};

} // KDNN

#endif
