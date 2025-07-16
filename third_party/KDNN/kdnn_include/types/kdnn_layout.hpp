/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: Enumeration of common data types which are supported in KDNN.
 * Author: KPL
 * Create: 2024-04-10
 * Notes: NA
 */

#ifndef KDNN_LAYOUT_HPP
#define KDNN_LAYOUT_HPP

namespace KDNN {

// Enumeration of common data types which are supported in KDNN.
enum class Layout {
    UNDEFINED = 0,
    A, // single row
    AB,
    BA,
    ABC,
    ACB,
    BAC,
    BCA,
    CAB,
    CBA,
    ABCD,
    ABDC,
    ACBD,
    ACDB,
    ADBC,
    ADCB,
    BACD,
    BCDA,
    CDAB,
    CDBA,
    DCAB,
    ABCDE,
    ABCED,
    ABDEC,
    ACBDE,
    ACDEB,
    ADECB,
    BACDE,
    BCDEA,
    CDEAB,
    CDEBA,
    DECAB,
    ROW_MAJOR = AB,
    COL_MAJOR = BA,
    NCHW = ABCD,
    NHWC = ACDB,
    NCDHW = ABCDE,
    NDHWC = ACDEB,
    OIHW = ABCD,
    HWIO = CDBA,
    HWOI = CDAB,
    OHWI = ACDB,
    OHWO = BCDA,
    IOHW = BACD,
};

} // KDNN

#endif // KDNN_LAYOUT_HPP
