#include <stdio.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::UnchangedShape;

REGISTER_OP("KPFusedEmbedding")
    .Input("weights: float")
    .Input("lookup: int64")
    .Input("dense_shape: int64")
    .Input("indices: int64")
    .Output("output: float")
    .Attr("combiner: int")

    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle temp;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(1), 1, &temp));
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(3), 2, &temp));
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(2), 1, &temp));
      ShapeHandle emb_var_shape;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 2, &emb_var_shape));

      DimensionHandle emb_size_dim = ctx->Dim(emb_var_shape, 1);
      DimensionHandle batch_dim = ctx->UnknownDim();

      ShapeHandle output_shape = ctx->MakeShape({batch_dim, emb_size_dim});
      ctx->set_output(0, output_shape);

      return OkStatus();
    });

REGISTER_OP("KPFusedEmbeddingWithHashBucket")
    .Input("lookup: string")
    .Input("weights: T_weight")
    .Attr("num_buckets: int >= 1")
    .Attr("combiner: int")
    .Attr("T_weight: {resource, float}")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle temp;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 1, &temp));
      DimensionHandle emb_size_dim = ctx->UnknownDim();
      DimensionHandle batch_dim = ctx->UnknownDim();

      ShapeHandle output_shape = ctx->MakeShape({batch_dim, emb_size_dim});
      ctx->set_output(0, output_shape);

      return OkStatus();
    });
    
REGISTER_OP("KPSoftmax")
    .Input("logits: T")
    .Output("softmax: T")
    .Attr("T: {float}")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });

}  // namespace tensorflow