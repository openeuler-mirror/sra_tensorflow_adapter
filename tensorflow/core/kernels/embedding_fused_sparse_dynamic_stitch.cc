/* Copyright 2025 The Huawei Technologies Co. Authors. All Rights Reserved.

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

#include <arm_neon.h>

#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;

class KPFusedSparseDynamicStitchOp : public OpKernel {
 public:
  explicit KPFusedSparseDynamicStitchOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    float* output;
    const Tensor& x = context->input(0);
    auto x_flat = x.flat<int64>();
    int64_t num_elems = x_flat.size();

    const int num_inputs = context->num_inputs();
    const int num_partitions = num_inputs - 1;
    int output_stride = 0;
    std::vector<const float*> variables(num_partitions);
    for (int i = 1; i < num_inputs; ++i) {
      if (i == 1) {
        const Tensor& input_tensor = context->input(i);
        if (input_tensor.shape().dims() == 2) {
          output_stride = input_tensor.shape().dim_size(1);
        }
      }
      variables[i - 1] = context->input(i).flat<float>().data();
    }

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({num_elems, output_stride}),
                                            &output_tensor));
    output = (float*)output_tensor->tensor_data().data();

    const size_t copy_size = output_stride * sizeof(float);
    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    const int64 cost_per_unit = 1000 * num_elems;
    auto work = [&](int start, int end) {
      const size_t copy_size = output_stride * sizeof(float);

      for (int i = start; i < end; ++i) {
        const int64_t global_id = x_flat(i);
        const int64_t table_id = global_id % num_partitions;
        const int64_t row_id = global_id / num_partitions;

        std::memcpy(output + i * output_stride,
                    variables[table_id] + row_id * output_stride, copy_size);
      }
    };

    Shard(worker_threads->num_threads, worker_threads->workers, num_elems,
          cost_per_unit, work);
  }
};

REGISTER_KERNEL_BUILDER(Name("KPFusedSparseDynamicStitch").Device(DEVICE_CPU),
                        KPFusedSparseDynamicStitchOp);
