/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Enumeration of common propagation types which are supported in KDNN.
 * Author: KPL
 * Create: 2024-07-19
 * Notes: NA
 */

#ifndef KDNN_PROPAGATION_TYPE_HPP
#define KDNN_PROPAGATION_TYPE_HPP

namespace KDNN {

// Enumeration of common propagation types which are supported in KDNN.
enum class Propagation {
    UNDEFINED = 0,
    FORWARD_TRAINING,
    FORWARD_INFERENCE,
    BACKWARD_DATA,
    BACKWARD_WEIGHTS,
    BACKWARD_BIAS,
    BACKWARD,
    FORWARD = FORWARD_TRAINING,
};

} // KDNN

#endif // KDNN_PROPAGATION_TYPE_HPP
