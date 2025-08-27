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

#include <vector>
#include <algorithm>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

class KPFusedSparseSelect : public OpKernel {
public:
  explicit KPFusedSparseSelect(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_a = context->input(0);
    const Tensor& input_b = context->input(1);
    const Tensor& input_c = context->input(2);

    auto a_flat = input_a.flat<int32_t>();
    auto b_flat = input_b.flat<int32_t>();
    auto c_flat = input_c.flat<int32_t>();
    VLOG(1) << "input_a shape: " << input_a.shape().DebugString();
    VLOG(1) << "input_b shape: " << input_b.shape().DebugString();
    VLOG(1) << "input_c shape: " << input_c.shape().DebugString();
    OP_REQUIRES(context, input_a.NumElements() == input_b.NumElements(),
                errors::InvalidArgument("Input num elements must match"));
    OP_REQUIRES(context, input_a.NumElements() == input_c.NumElements(),
                errors::InvalidArgument("Input num elements must match"));
    auto N = input_a.NumElements();

    Eigen::TensorMap<Eigen::Tensor<const int32_t, 2, Eigen::RowMajor>> a_reshaped_tensor(a_flat.data(), N, 1);
    Eigen::TensorMap<Eigen::Tensor<const int32_t, 2, Eigen::RowMajor>> b_reshaped_tensor(b_flat.data(), N, 1);
    Eigen::TensorMap<Eigen::Tensor<const int32_t, 2, Eigen::RowMajor>> c_reshaped_tensor(c_flat.data(), N, 1);

    Tensor* output_x = nullptr;
    Tensor* output_y = nullptr;
    Tensor* output_w = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({N, 1}), &output_x));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({N, 1}), &output_y));
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({N, 2}), &output_w));

    auto out_x = output_x->matrix<float>();  // [N,1]
    auto out_y = output_y->matrix<float>();  // [N,1]
    auto out_w = output_w->matrix<float>();  // [N,2]

    auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
    const int64 cost_per_unit = 10;
    
    auto work = [&](int64 start, int64 end) {
      for (int64 i = start; i < end; i++) {
        float a_greater = (a_reshaped_tensor(i) > 0) ? 1.0f : 0.0f;
        float select_2412 = (b_reshaped_tensor(i) == 4563) ? 1.0f : a_greater;
        float select_2415 = (b_reshaped_tensor(i) == 10831) ? 1.0f : select_2412;
        float sub_out = 1.0f - select_2415;
        out_x(i, 0) = 0.0f; 
        out_y(i, 0) = sub_out;
        out_w(i, 0) = select_2415;
        out_w(i, 1) = 1.0f;
      }
    };
    Shard(worker_threads->num_threads, worker_threads->workers, N, cost_per_unit, work);
    
    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> map_output_x(
        output_x->flat<float>().data(),
        output_x->dim_size(0),
        output_x->dim_size(1)
    );

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> map_output_y(
        output_y->flat<float>().data(),
        output_y->dim_size(0),
        output_y->dim_size(1)
    );

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> map_output_w(
        output_w->flat<float>().data(),
        output_w->dim_size(0),
        output_w->dim_size(1)
    );
  }
};

REGISTER_KERNEL_BUILDER(Name("KPFusedSparseSelect").Device(DEVICE_CPU),
                        KPFusedSparseSelect);