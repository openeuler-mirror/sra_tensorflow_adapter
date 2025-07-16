/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Local Response Normalization
 * Author: KPL
 * Create: 2024-07-29
 * Notes: NA
 */

#ifndef KDNN_LRN_HPP
#define KDNN_LRN_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

enum class LRNormAlgorithmKind : std::uint32_t {
    WITHIN         = 0x0U,
    ACROSS         = 0x1U,
};

namespace Detail {

class LocalResponseNormalizationLayerFWDImpl;
class LocalResponseNormalizationLayerBWDImpl;

} // Detail

class KDNN_API_PUBLIC LocalResponseNormalizationLayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo, float alpha, float beta, float k,
        SizeType localSize, LRNormAlgorithmKind algorithm) noexcept;
    LocalResponseNormalizationLayerFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo, float alpha, float beta,
        float k, SizeType localSize, LRNormAlgorithmKind algorithm) noexcept(false);
    LocalResponseNormalizationLayerFWD(const LocalResponseNormalizationLayerFWD &other) noexcept(false);
    LocalResponseNormalizationLayerFWD(LocalResponseNormalizationLayerFWD &&other) noexcept;
    LocalResponseNormalizationLayerFWD& operator=(const LocalResponseNormalizationLayerFWD &other) noexcept(false);
    LocalResponseNormalizationLayerFWD& operator=(LocalResponseNormalizationLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~LocalResponseNormalizationLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::LocalResponseNormalizationLayerFWDImpl> pImpl;
};

class KDNN_API_PUBLIC LocalResponseNormalizationLayerBWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo,
        const TensorInfo &diffDstInfo, const TensorInfo &diffSrcInfo,
        float alpha, float beta, float k,
        SizeType localSize, LRNormAlgorithmKind algorithm) noexcept;
    LocalResponseNormalizationLayerBWD(const TensorInfo &srcInfo,
        const TensorInfo &diffDstInfo, const TensorInfo &diffSrcInfo,
        float alpha, float beta, float k,
        SizeType localSize, LRNormAlgorithmKind algorithm) noexcept(false);
    LocalResponseNormalizationLayerBWD(const LocalResponseNormalizationLayerBWD &other) noexcept(false);
    LocalResponseNormalizationLayerBWD(LocalResponseNormalizationLayerBWD &&other) noexcept;
    LocalResponseNormalizationLayerBWD& operator=(const LocalResponseNormalizationLayerBWD &other) noexcept(false);
    LocalResponseNormalizationLayerBWD& operator=(LocalResponseNormalizationLayerBWD &&other) noexcept;
    void Run(const void *src, const void *diffDst,
            void *diffSrc) const noexcept(false);
    ~LocalResponseNormalizationLayerBWD() noexcept;
private:
    std::unique_ptr<Detail::LocalResponseNormalizationLayerBWDImpl> pImpl;
};

} // KDNN

#endif // KDNN_LRN_HPP
