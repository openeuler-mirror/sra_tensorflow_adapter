// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2010 Konstantinos Margaritis <markos@freevec.org>
// Heavily based on Gael's SSE version.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef KUNPENG_TOOLS_H
#define KUNPENG_TOOLS_H

#include "third_party/eigen3/Eigen/Core"
 
namespace Eigen {
namespace internal {
template <>
EIGEN_STRONG_INLINE Packet2d pfloor<Packet2d>(const Packet2d &a)
{
    const Packet2d cst_1 = pset1<Packet2d>(1.0);
    /* perform a floorf */
    const Packet2d tmp = vcvtq_f64_s64(vcvtq_s64_f64(a));
 
    /* if greater, substract 1 */
    uint64x2_t mask = vcgtq_f64(tmp, a);
    mask = vandq_u64(mask, vreinterpretq_u64_f64(cst_1));
    return vsubq_f64(tmp, vreinterpretq_f64_u64(mask));
}
}  // namespace internal
}  // namespace Eigen

#endif // KUNPENG_TOOLS_H