#ifndef TF_BINARYOP_H
#define TF_BINARYOP_H

#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/util/work_sharder.h"
#include "tools.h"
#include "third_party/ktfop/include/ktfop.h"
#include <chrono>
#include <iostream>
using namespace ktfop;

namespace tensorflow {
#define REGISTER_BINARY_FUNC(tf_func, scalar_type, ock_in_type, ock_out_type, func)                               \
    template <> class BinaryOp<CPUDevice, tf_func<scalar_type>> : public BinaryOpShared {                         \
    public:                                                                                                       \
        typedef CPUDevice Device;                                                                                 \
        typedef typename tf_func<scalar_type> Functor;                                                            \
        typedef typename Functor::in_type Tin;                                                                    \
        typedef typename Functor::out_type Tout;                                                                  \
        explicit BinaryOp(OpKernelConstruction *ctx)                                                              \
            : BinaryOpShared(ctx, DataTypeToEnum<Tout>::v(), DataTypeToEnum<Tin>::v())                            \
        {}                                                                                                        \
        void Compute(OpKernelContext *ctx) override                                                               \
        {                                                                                                         \
            BinaryOpState state(ctx);                                                                             \
            auto &bcast = state.bcast;                                                                            \
            const Device &eigen_device = ctx->eigen_device<Device>();                                             \
            auto &in0 = state.in0;                                                                                \
            auto &in1 = state.in1;                                                                                \
            Tensor *out = state.out;                                                                              \
            if (!bcast.IsValid()) {                                                                               \
                if (ctx->status().ok()) {                                                                         \
                    if (state.result) {                                                                           \
                        functor::SetOneFunctor<Device, bool>()(eigen_device, out->flat<bool>());                  \
                    } else {                                                                                      \
                        functor::SetZeroFunctor<Device, bool>()(eigen_device, out->flat<bool>());                 \
                    }                                                                                             \
                }                                                                                                 \
                return;                                                                                           \
            }                                                                                                     \
            if (state.out_num_elements == 0) {                                                                    \
                return;                                                                                           \
            }                                                                                                     \
            auto input0 = in0.flat<Tin>();                                                                        \
            auto input1 = in1.flat<Tin>();                                                                        \
            auto output_flat = out->flat<Tout>();                                                                 \
            ock_in_type *input0_data =                                                                            \
                const_cast<ock_in_type *>(reinterpret_cast<const ock_in_type *>(input0.data()));                  \
            ock_in_type *input1_data =                                                                            \
                const_cast<ock_in_type *>(reinterpret_cast<const ock_in_type *>(input1.data()));                  \
            ock_out_type *output_data =                                                                           \
                const_cast<ock_out_type *>(reinterpret_cast<const ock_out_type *>(output_flat.data()));           \
            const int ndims = state.ndims;                                                                        \
            bool error = false;                                                                                   \
            bool * const error_ptr = Functor::has_errors ? &error : nullptr;                                      \
            const int PacketSize = Eigen::internal::packet_traits<Tin>::size;                                     \
            const int size = output_flat.size();                                                                  \
            int n = size / PacketSize;                                                                            \
            if (ndims <= 1) {                                                                                     \
                if (state.in1_num_elements == 1 && state.in0_num_elements != 1) {                                 \
                    auto work = [&](int64_t start, int64_t end) {                                                 \
                        func(input0_data + start, input1_data[0], output_data + start, end - start);              \
                    };                                                                                            \
                    auto *worker_threads = ctx->device()->tensorflow_cpu_worker_threads();                        \
                    Shard(worker_threads->num_threads, worker_threads->workers, size,                             \
                        8, work);                                              \
                } else if (state.in0_num_elements == 1 && state.in1_num_elements != 1) {                          \
                    auto work = [&](int64_t start, int64_t end) {                                                 \
                        func(input0_data[0], input1_data + start, output_data + start, end - start);              \
                    };                                                                                            \
                    auto *worker_threads = ctx->device()->tensorflow_cpu_worker_threads();                        \
                    Shard(worker_threads->num_threads, worker_threads->workers, size,                             \
                        8, work);                                              \
                } else {                                                                                          \
                    auto work = [&](int64_t start, int64_t end) {                                                 \
                        func(input0_data + start, input1_data + start, output_data + start, end - start);         \
                    };                                                                                            \
                    auto *worker_threads = ctx->device()->tensorflow_cpu_worker_threads();                        \
                    Shard(worker_threads->num_threads, worker_threads->workers, size,                             \
                        8, work);                                              \
                }                                                                                                 \
            } else if (ndims == 2) {                                                                              \
                functor::BinaryFunctor<Device, Functor, 2>().BCast(eigen_device,                                  \
                    out->shaped<Tout, 2>(bcast.result_shape()), in0.template shaped<Tin, 2>(bcast.x_reshape()),   \
                    BCast::ToIndexArray<2>(bcast.x_bcast()), in1.template shaped<Tin, 2>(bcast.y_reshape()),      \
                    BCast::ToIndexArray<2>(bcast.y_bcast()), error_ptr);                                          \
            } else if (ndims == 3) {                                                                              \
                functor::BinaryFunctor<Device, Functor, 3>().BCast(eigen_device,                                  \
                    out->shaped<Tout, 3>(bcast.result_shape()), in0.template shaped<Tin, 3>(bcast.x_reshape()),   \
                    BCast::ToIndexArray<3>(bcast.x_bcast()), in1.template shaped<Tin, 3>(bcast.y_reshape()),      \
                    BCast::ToIndexArray<3>(bcast.y_bcast()), error_ptr);                                          \
            } else if (ndims == 4) {                                                                              \
                functor::BinaryFunctor<Device, Functor, 4>().BCast(eigen_device,                                  \
                    out->shaped<Tout, 4>(bcast.result_shape()), in0.template shaped<Tin, 4>(bcast.x_reshape()),   \
                    BCast::ToIndexArray<4>(bcast.x_bcast()), in1.template shaped<Tin, 4>(bcast.y_reshape()),      \
                    BCast::ToIndexArray<4>(bcast.y_bcast()), error_ptr);                                          \
            } else if (ndims == 5) {                                                                              \
                functor::BinaryFunctor<Device, Functor, 5>().BCast(eigen_device,                                  \
                    out->shaped<Tout, 5>(bcast.result_shape()), in0.template shaped<Tin, 5>(bcast.x_reshape()),   \
                    BCast::ToIndexArray<5>(bcast.x_bcast()), in1.template shaped<Tin, 5>(bcast.y_reshape()),      \
                    BCast::ToIndexArray<5>(bcast.y_bcast()), error_ptr);                                          \
            } else {                                                                                              \
                SetUnimplementedError(ctx);                                                                       \
            }                                                                                                     \
            if (Functor::has_errors && error) {                                                                   \
                SetComputeError(ctx);                                                                             \
            }                                                                                                     \
        }                                                                                                         \
    };

REGISTER_BINARY_FUNC(functor::floor_fmod, float, float, float, FloorMod);
REGISTER_BINARY_FUNC(functor::floor_fmod, double, double, double, FloorMod);
REGISTER_BINARY_FUNC(functor::less, int64, int64_t, bool, Less);
REGISTER_BINARY_FUNC(functor::less, int32, int32_t, bool, Less);
REGISTER_BINARY_FUNC(functor::greater, int64, int64_t, bool, Greater);
REGISTER_BINARY_FUNC(functor::greater, int32, int32_t, bool, Greater);
} // namespace tensorflow
#endif // TF_BINARYOP_H