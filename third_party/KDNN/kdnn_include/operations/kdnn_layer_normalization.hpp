/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Normalization
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_LAYER_NORMALIZATION_HPP
#define KDNN_LAYER_NORMALIZATION_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

namespace Detail {

class NormalizationLayerFWDImpl;
class NormalizationLayerBWDImpl;

} // Detail

class KDNN_API_PUBLIC NormalizationLayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &statsInfo,
	                            const TensorInfo &scaleshiftInfo, const TensorInfo &dstInfo,
                                NormalizationFlags flags) noexcept;
    NormalizationLayerFWD(const TensorInfo &srcInfo, const TensorInfo &statsInfo, const TensorInfo &scaleshiftInfo,
                          const TensorInfo &dstInfo, NormalizationFlags flags) noexcept(false);
    NormalizationLayerFWD(const NormalizationLayerFWD &other) noexcept(false);
    NormalizationLayerFWD(NormalizationLayerFWD &&other) noexcept;
    NormalizationLayerFWD& operator=(const NormalizationLayerFWD &other) noexcept(false);
    NormalizationLayerFWD& operator=(NormalizationLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst, const void *scale, const void *shift, float *mean, float *variance,
             bool saveStats, const float eps) const noexcept(false);
    ~NormalizationLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::NormalizationLayerFWDImpl> pImpl;
};

class KDNN_API_PUBLIC NormalizationLayerBWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &statInfo,
        const TensorInfo &diffSrcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &scaleShiftInfo, const TensorInfo &diffscaleShiftInfo,
        NormalizationFlags flags) noexcept;
    NormalizationLayerBWD(const TensorInfo &srcInfo, const TensorInfo &statInfo,
        const TensorInfo &diffSrcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &scaleShiftInfo, const TensorInfo &diffscaleShiftInfo,
        NormalizationFlags flags) noexcept(false);
    NormalizationLayerBWD(const NormalizationLayerBWD &other) noexcept(false);
    NormalizationLayerBWD(NormalizationLayerBWD &&other) noexcept;
    NormalizationLayerBWD& operator=(const NormalizationLayerBWD &other) noexcept(false);
    NormalizationLayerBWD& operator=(NormalizationLayerBWD &&other) noexcept;
    void Run(const void *src, const float *mean, const float *variance, const void *diffDst,
            const void *scale, void *diffSrc, void *diffScale,
            void *diffShift, float eps) const noexcept(false);
    ~NormalizationLayerBWD() noexcept;
private:
    std::unique_ptr<Detail::NormalizationLayerBWDImpl> pImpl;
};

} // KDNN

#endif // KDNN_LAYER_NORMALIZATION_HPP
