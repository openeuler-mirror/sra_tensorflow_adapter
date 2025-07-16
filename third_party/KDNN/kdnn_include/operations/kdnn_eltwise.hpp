/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: mathematical correlation function
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_ELTWISE_HPP
#define KDNN_ELTWISE_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {
#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

// supported functions
enum class ActivationFunction {
    UNIMPLEMENTED,
    ABS,
    EXP,
    LINEAR,
    LOG,
    RELU,
    ROUND,
    SQRT,
    SQUARE,
    TANH,
    SIGMOID,
    POW,
    CLIP,
    CLIP_V2,
    HARDSIGMOID,
    HARDSWISH,
    EXP_DST,
    LOGISTIC_DST,
    TANH_DST,
    ELU,
    GELU_TANH,
    CLIP_V2_DST,
    ELU_DST,
    RELU_DST,
    SQRT_DST,
    SWISH,
    SRELU,
    GELU_ERF,
    MISH
};

namespace Detail {

class ActivationLayerImplFWD;
class ActivationLayerImplBWD;

} // Detail

class KDNN_API_PUBLIC ActivationLayerFWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
        ActivationFunction kind, float alpha = 0.0f, float beta = 0.0f) noexcept;
    ActivationLayerFWD(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
        ActivationFunction kind, float alpha = 0.0f, float beta = 0.0f) noexcept(false);
    ActivationLayerFWD(const ActivationLayerFWD &other) noexcept(false);
    ActivationLayerFWD(ActivationLayerFWD &&other) noexcept;
    ActivationLayerFWD& operator=(const ActivationLayerFWD &other) noexcept(false);
    ActivationLayerFWD& operator=(ActivationLayerFWD &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~ActivationLayerFWD() noexcept;
private:
    std::unique_ptr<Detail::ActivationLayerImplFWD> pImpl;
};

class KDNN_API_PUBLIC ActivationLayerBWD final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &dsInfo, const TensorInfo &ddInfo,
        const TensorInfo &srcInfo, ActivationFunction kind, float alpha = 0.0f,
        float beta = 0.0f) noexcept;
    ActivationLayerBWD(const TensorInfo &dsInfo, const TensorInfo &ddInfo,
        const TensorInfo &srcInfo, ActivationFunction kind, float alpha = 0.0f,
        float beta = 0.0f) noexcept(false);
    ActivationLayerBWD(const ActivationLayerBWD &other) noexcept(false);
    ActivationLayerBWD(ActivationLayerBWD &&other) noexcept;
    ActivationLayerBWD& operator=(const ActivationLayerBWD &other) noexcept(false);
    ActivationLayerBWD& operator=(ActivationLayerBWD &&other) noexcept;
    void Run(void *ds, const void *dd, const void *src) const noexcept(false);
    ~ActivationLayerBWD() noexcept;
private:
    std::unique_ptr<Detail::ActivationLayerImplBWD> pImpl;
};

} // KDNN

#endif // KDNN_ELTWISE_HPP
