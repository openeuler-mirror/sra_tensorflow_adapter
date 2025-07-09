/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Reduction
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_REDUCTION_HPP
#define KDNN_REDUCTION_HPP

#include <memory>

#include "types/kdnn_tensor_info.hpp"
#include "service/kdnn_err_codes.hpp"

namespace KDNN {

// supported functions
enum class ReductionFunction {
    UNIMPLEMENTED,
    MAX,
    MIN,
    SUM,
    MUL,
    MEAN,
    NORM_LP_MAX,
    NORM_LP_SUM,
    NORM_LP_POWER_P_MAX,
    NORM_LP_POWER_P_SUM,
};

namespace Detail {

class ReductionLayerImpl;

} // Detail

class KDNN_API_PUBLIC ReductionLayer final {
public:
    using SizeType = ::KDNN::SizeType;
    static Status ValidateInput(const TensorInfo &srcInfo, const TensorInfo &dstInfo,
	    const ReductionFunction& reductionAlg, float p, float eps) noexcept;
    ReductionLayer(const TensorInfo &srcInfo, const TensorInfo &dstInfo, ReductionFunction kind,
        float p = 1, float eps = 0) noexcept(false);
    ReductionLayer(const ReductionLayer &other) noexcept(false);
    ReductionLayer(ReductionLayer &&other) noexcept;
    ReductionLayer& operator=(const ReductionLayer &other) noexcept(false);
    ReductionLayer& operator=(ReductionLayer &&other) noexcept;
    void Run(const void *src, void *dst) const noexcept(false);
    ~ReductionLayer() noexcept;
private:
    std::unique_ptr<Detail::ReductionLayerImpl> pImpl;
};

} // KDNN

#endif // KDNN_REDUCTION_HPP
