/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Group Normalization
 * Author: KPL
 * Create: 2024-10-10
 * Notes: NA
 */

#ifndef KDNN_GROUP_NORMALIZATION_HPP
#define KDNN_GROUP_NORMALIZATION_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

namespace Detail {

class GroupNormalizationLayerFWDImpl;
class GroupNormalizationLayerBWDImpl;

} // Detail

class KDNN_API_PUBLIC GroupNormalizationLayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &scaleshiftInfo, SizeType groupInfo,
        const TensorInfo &dstInfo, NormalizationFlags flags) noexcept;
    GroupNormalizationLayerFWD(const TensorInfo &srcInfo, const TensorInfo &scaleshiftInfo, SizeType groupSize,
                          const TensorInfo &dstInfo, NormalizationFlags flags) noexcept(false);
    GroupNormalizationLayerFWD(const GroupNormalizationLayerFWD &other) noexcept(false);
    GroupNormalizationLayerFWD(GroupNormalizationLayerFWD &&other) noexcept;
    GroupNormalizationLayerFWD& operator=(const GroupNormalizationLayerFWD &other) noexcept(false);
    GroupNormalizationLayerFWD& operator=(GroupNormalizationLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst, const void *scale, const void *shift, float *mean, float *variance,
             bool saveStats, const float eps) const noexcept(false);
    ~GroupNormalizationLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::GroupNormalizationLayerFWDImpl> pImpl;
};

class KDNN_API_PUBLIC GroupNormalizationLayerBWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &statInfo,
        const TensorInfo &diffSrcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &scaleShiftInfo, const TensorInfo &diffscaleShiftInfo,
        NormalizationFlags flags) noexcept;
    GroupNormalizationLayerBWD(const TensorInfo &srcInfo, const TensorInfo &statInfo,
        const TensorInfo &diffSrcInfo, const TensorInfo &diffDstInfo,
        const TensorInfo &scaleShiftInfo, const TensorInfo &diffscaleShiftInfo,
        NormalizationFlags flags) noexcept(false);
    GroupNormalizationLayerBWD(const GroupNormalizationLayerBWD &other) noexcept(false);
    GroupNormalizationLayerBWD(GroupNormalizationLayerBWD &&other) noexcept;
    GroupNormalizationLayerBWD& operator=(const GroupNormalizationLayerBWD &other) noexcept(false);
    GroupNormalizationLayerBWD& operator=(GroupNormalizationLayerBWD &&other) noexcept;
    void Run(const void *src, const float *mean, const float *variance, const void *diffDst,
            const void *scale, void *diffSrc, void *diffScale,
            void *diffShift, float eps) const noexcept(false);
    ~GroupNormalizationLayerBWD() noexcept;
private:
    std::unique_ptr<Detail::GroupNormalizationLayerBWDImpl> pImpl;
};

} // KDNN

#endif // KDNN_GROUP_NORMALIZATION_HPP