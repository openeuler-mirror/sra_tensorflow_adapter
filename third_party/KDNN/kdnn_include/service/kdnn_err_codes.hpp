/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Error codes
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_ERR_CODES
#define KDNN_ERR_CODES

namespace KDNN {

enum class Status {
    SUCCESS = 0,
    UNSUPPORTED,
    BAD_ARGUMENTS
};

} // KDNN

#endif // KDNN_ERR_CODES
