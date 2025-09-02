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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;

class KPFusedGather : public OpKernel {
 public:
  explicit KPFusedGather(OpKernelConstruction* context) : OpKernel(context) { }

  void Compute(OpKernelContext* context) override {
    const Tensor& data = context->input(0);
    const Tensor& slice_input = context->input(1);
    const Tensor& begin = context->input(2);

    OP_REQUIRES(context, slice_input.dims() == 2, errors::Internal("slice_input dims must == 2"));
    OP_REQUIRES(context, data.dims() == 2, errors::Internal("identity dims must == 2"));
    OP_REQUIRES(context, data.dim_size(1) == 12, errors::Internal("identity dim size must == [n, 12]"));

    VLOG(1) << "Input identity shape: " << data.shape().DebugString();
    VLOG(1) << "Input slice_input shape: " << slice_input.shape().DebugString();
    VLOG(1) << "Input slice_input: " << slice_input.SummarizeValue(1000);
    VLOG(1) << "Input begin value: " << begin.SummarizeValue(10);

    int32 col = begin.flat<int32>().data()[1];
    OP_REQUIRES(context, col < slice_input.dim_size(1), errors::Internal("begin[1] must < slice_input.dim_size(1)"));
    auto data_mat = data.matrix<float>();
    auto slice_input_mat = slice_input.matrix<int64>();

    VLOG(1) << "Column index from begin: " << col;

    std::vector<int64_t> unique_values;
    std::vector<int32_t> indices(slice_input.dim_size(0));
    std::unordered_map<int64_t, int32_t> value_to_index;
    int current_index = 0;
    for (int64_t i = 0; i < slice_input.dim_size(0); ++i) {
        auto it = value_to_index.find(slice_input_mat(i, col));
        if (it == value_to_index.end()) {
            value_to_index[slice_input_mat(i, col)] = current_index;
            unique_values.push_back(slice_input_mat(i, col));
            indices[i] = current_index;
            current_index++;
        } else {
            indices[i] = it->second;
        }
    }

    Tensor* out_shape = nullptr;
    Tensor* out_indices = nullptr;
    Tensor* out_data = nullptr;

    OP_REQUIRES_OK(context,
                   context->allocate_output(
                   0, TensorShape({unique_values.size()}), &out_shape));
    std::memcpy(out_shape->data(), unique_values.data(), unique_values.size() * sizeof(int64_t));

    OP_REQUIRES_OK(context,
                   context->allocate_output(
                   1, TensorShape({static_cast<int32>(indices.size())}), &out_indices));
    std::memcpy(out_indices->data(), indices.data(), indices.size() * sizeof(int32_t));

    OP_REQUIRES_OK(context,
                   context->allocate_output(
                   2, TensorShape({unique_values.size(), data.dim_size(1)}), &out_data));
    auto output_data = out_data->matrix<float>();

    int64_t data_row = data.dim_size(0);
    int64_t cols = data.dim_size(1);
    for (int64_t cur_row = 0; cur_row < unique_values.size(); ++cur_row) {
        int64_t idx = unique_values[cur_row];
        OP_REQUIRES(context, idx < data_row, errors::Internal("idx must < data_row"));
        const float* src = data_mat.data() + idx * cols;
        float* dst = output_data.data() + cur_row * cols;
        std::memcpy(dst, src, cols * sizeof(float));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("KPFusedGather").Device(DEVICE_CPU),
                        KPFusedGather);