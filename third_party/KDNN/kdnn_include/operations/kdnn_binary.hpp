/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: binary correlation function
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_BINARY_HPP
#define KDNN_BINARY_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

enum class BinaryFunction {
    UNIMPLEMENTED,
    ADD,
    DIV,
    MAX,
    MIN,
    MUL,
    SUB,
    GE,
    GT,
    LE,
    LT,
    EQ,
    NE
};

namespace Detail {

class BinaryLayerImpl;

} // Detail

class KDNN_API_PUBLIC BinaryLayer final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &src0Info, const TensorInfo &src1Info,
        const TensorInfo &dstInfo, BinaryFunction op) noexcept;
    BinaryLayer(const TensorInfo &src0Info, const TensorInfo &src1Info,
        const TensorInfo &dstInfo, BinaryFunction op) noexcept(false);
    BinaryLayer(const BinaryLayer &other) noexcept(false);
    BinaryLayer(BinaryLayer &&other) noexcept;
    BinaryLayer& operator=(const BinaryLayer &other) noexcept(false);
    BinaryLayer& operator=(BinaryLayer &&other) noexcept;
    void Run(const void *src0, const void *src1, void *dst,
        const float scale0 = 1.0f, const float scale1 = 1.0f) const noexcept(false);
    ~BinaryLayer() noexcept;
private:
    std::unique_ptr<Detail::BinaryLayerImpl> pImpl;
};

} // KDNN

#endif // KDNN_BINARY_HPP
