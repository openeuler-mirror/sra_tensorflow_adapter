/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Batch Normalization
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_BATCH_NORMALIZATION_HPP
#define KDNN_BATCH_NORMALIZATION_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "types/kdnn_normalization.hpp"
#include "service/kdnn_err_codes.hpp"
#include "types/kdnn_propagation_type.hpp"

namespace KDNN {

namespace Detail {

class BatchNormalizationLayerFWDImpl;
class BatchNormalizationLayerBWDImpl;

} // Detail

class KDNN_API_PUBLIC BatchNormalizationLayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const Propagation &propKind, const TensorInfo &srcInfo, const TensorInfo &statsInfo,
        const TensorInfo &scaleShiftInfo, const TensorInfo &dstInfo,
        NormalizationFlags flags) noexcept;
    BatchNormalizationLayerFWD(const Propagation &propKind, const TensorInfo &srcInfo, const TensorInfo &statsInfo,
        const TensorInfo &scaleShiftInfo, const TensorInfo &dstInfo, NormalizationFlags flags) noexcept(false);
    BatchNormalizationLayerFWD(const BatchNormalizationLayerFWD &other) noexcept(false);
    BatchNormalizationLayerFWD(BatchNormalizationLayerFWD &&other) noexcept;
    BatchNormalizationLayerFWD& operator=(const BatchNormalizationLayerFWD &other) noexcept(false);
    BatchNormalizationLayerFWD& operator=(BatchNormalizationLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst, const float *scale, const float *shift, float *mean, float *variance,
        bool saveStats, const float eps, void *ws = nullptr) const noexcept(false);
    ~BatchNormalizationLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::BatchNormalizationLayerFWDImpl> pImpl;
};

class KDNN_API_PUBLIC BatchNormalizationLayerBWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const Propagation &propKind, const TensorInfo &srcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &statsInfo, const TensorInfo &scaleShiftInfo, const TensorInfo &diffSrcInfo,
        const TensorInfo &diffScaleShiftInfo, NormalizationFlags flags) noexcept;
    BatchNormalizationLayerBWD(const Propagation &propKind, const TensorInfo &srcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &statsInfo, const TensorInfo &scaleShiftInfo, const TensorInfo &diffSrcInfo,
        const TensorInfo &diffScaleShiftInfo, NormalizationFlags flags) noexcept(false);
    BatchNormalizationLayerBWD(const BatchNormalizationLayerBWD &other) noexcept(false);
    BatchNormalizationLayerBWD(BatchNormalizationLayerBWD &&other) noexcept;
    BatchNormalizationLayerBWD& operator=(const BatchNormalizationLayerBWD &other) noexcept(false);
    BatchNormalizationLayerBWD& operator=(BatchNormalizationLayerBWD &&other) noexcept;
    void Run(const void *src, const void *diffDst, const float *mean, const float *variance, const float *scale,
        void *diffSrc, float *diffScale, float *diffShift, const float eps,
        const void *ws = nullptr) const noexcept(false);
    ~BatchNormalizationLayerBWD() noexcept;
private:
    std::unique_ptr<Detail::BatchNormalizationLayerBWDImpl> pImpl;
};

} // KDNN

#endif // KDNN_BATCH_NORMALIZATION_HPP
