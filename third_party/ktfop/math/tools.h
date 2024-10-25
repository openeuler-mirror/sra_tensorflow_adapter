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