/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "third_party/ktfop/math/BinaryOp.h"

namespace tensorflow {
REGISTER9(BinaryOp, CPU, "Greater", functor::greater, float, Eigen::half,
          double, int32, int64, uint8, int8, int16, bfloat16);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER7(BinaryOp, GPU, "Greater", functor::greater, float, Eigen::half,
          double, int64, uint8, int8, int16);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Greater")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::greater<int32>>);
#endif
#ifdef TENSORFLOW_USE_SYCL
REGISTER2(BinaryOp, SYCL, "Greater", functor::greater, float, double);

REGISTER_KERNEL_BUILDER(Name("Greater")
                            .Device(DEVICE_SYCL)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::greater<int32>>);
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow