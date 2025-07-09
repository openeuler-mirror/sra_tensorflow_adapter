/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Enumeration of common data types which are supported in KDNN.
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_NORMALIZATION_HPP
#define KDNN_NORMALIZATION_HPP

#include "types/kdnn_data_type.hpp"

namespace KDNN {

enum class KDNN_API_PUBLIC NormalizationFlags : std::uint32_t {
    NONE               = 0x0U,
    USE_GLOBAL_STATS   = 0x1U,
    USE_SCALE          = 0x2U,
    USE_SHIFT          = 0x4U,
    FUSE_NORM_RELU     = 0x8U
};

KDNN_API_PUBLIC NormalizationFlags operator & (const NormalizationFlags &lhs, const NormalizationFlags &rhs);
KDNN_API_PUBLIC NormalizationFlags operator | (const NormalizationFlags &lhs, const NormalizationFlags &rhs);
KDNN_API_PUBLIC NormalizationFlags &operator &= (NormalizationFlags &lhs, const NormalizationFlags &rhs);
KDNN_API_PUBLIC NormalizationFlags &operator |= (NormalizationFlags &lhs, const NormalizationFlags &rhs);

} // namespace KDNN

#endif // KDNN_NORMALIZATION_HPP
