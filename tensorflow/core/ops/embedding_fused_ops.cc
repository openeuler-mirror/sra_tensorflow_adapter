/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <stdio.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::UnchangedShape;

REGISTER_OP("KPFusedSparseSegmentReduce")
    .Input("data: float")
    .Input("indices: Tidx")
    .Input("slice_input: int64")
    .Input("begin: int32")
    .Input("end: int32")
    .Input("strides: int32")
    .Attr("combiner: int = 1")  // 0 for SUM, 1 for MEAN
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Output("output: float")
    .Output("slice_output: int32")
    .SetShapeFn(shape_inference::UnknownShape);

REGISTER_OP("KPFusedSparseConcat")
    .Input("shape: int64")
    .Input("pooling: float")
    .Input("pooling_rows: int32")
    .Output("output0: int32")
    .Output("output1: int32")
    .SetShapeFn(shape_inference::UnknownShape);

}  // namespace tensorflow