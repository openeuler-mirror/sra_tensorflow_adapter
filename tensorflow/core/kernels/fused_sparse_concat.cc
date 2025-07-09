/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <iostream>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

class KPFusedSparseConcatOP : public OpKernel {
 public:
  explicit KPFusedSparseConcatOP(OpKernelConstruction* context) : OpKernel(context) { }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& shape = context->input(0);
    const Tensor& pooling = context->input(1);
    const Tensor& pooling_rows = context->input(2);

    OP_REQUIRES(context, shape.dims() == 1, errors::Internal("shape dims must == 1"));
    OP_REQUIRES(context, shape.NumElements() >= 1, errors::Internal("shape NumElements must >= 1"));
    OP_REQUIRES(context, pooling.dims() == 2, errors::Internal("pooling dims must == 2"));
    OP_REQUIRES(context, pooling.dim_size(1) == 10, errors::Internal("pooling dim_size(1) == 10"));
    OP_REQUIRES(context, pooling_rows.dims() == 0, errors::Internal("pooling_rows must be a scalar"));

    int padding_rows = static_cast<int32>(shape.flat<int64>()(0)) - pooling_rows.scalar<int32>()();
    OP_REQUIRES(context, padding_rows >= 0, errors::Internal("padding_rows must >= 0"));
    Tensor* output0 = nullptr;
    Tensor* output1 = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}),
                                            &output0));

    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({}),
                                            &output1));
    output0->scalar<int32>()() = padding_rows;

    int first_dims = padding_rows + pooling.dim_size(0);
    int second_dims = pooling.dim_size(1);
    OP_REQUIRES(context, first_dims * second_dims % 1510 == 0, errors::Internal("padding cannot reshape to [-1, 1510]"));
    output1->scalar<int32>()() = first_dims * second_dims / 1510;
  }
};

REGISTER_KERNEL_BUILDER(Name("KPFusedSparseConcat").Device(DEVICE_CPU),
                        KPFusedSparseConcatOP);