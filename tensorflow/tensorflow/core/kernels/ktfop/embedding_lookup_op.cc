#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/bounds_check.h"

#include "ktfop.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

class KPFusedEmbeddingOp : public OpKernel {
public:
  explicit KPFusedEmbeddingOp(OpKernelConstruction* context)
           : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("combiner", &combiner_));
    node_name = context->def().name();
  }

  ~KPFusedEmbeddingOp() {}

  void Compute(OpKernelContext* context) override {
    float *weight;
    const Tensor* weight_tensor = &context->input(0);

    if (weight_tensor->dtype() == DT_RESOURCE) {
      Var* variable;
      OP_REQUIRES_OK(context,
                     LookupResource(context, HandleFromInput(context, 0), 
                                    &variable));
      core::ScopedUnref s(variable);
      weight_tensor = variable->tensor();
      OP_REQUIRES(context, weight_tensor->dtype() == DT_FLOAT,
                  errors::InvalidArgument("Expect float weight in ",
                                          node_name));
    }

    weight = (float *)weight_tensor->tensor_data().data();
    
    const Tensor& input_tensor = context->input(1);
    int64 *input = (int64 *)input_tensor.tensor_data().data();
    const Tensor& shape_tensor = context->input(2);
    int64 *shape = (int64 *)shape_tensor.tensor_data().data();

    OP_REQUIRES(context, (shape_tensor.dims() == 1),
                errors::InvalidArgument("Shape tensor is not valid (dims != 1)"));
    OP_REQUIRES(context, (shape_tensor.dim_size(0) >= 2),
                errors::InvalidArgument("Shape tensor is not valid (dim_size(0) < 2)"));
    
    int64 input_size = 1;
    for (int i = 0; i < input_tensor.dims(); ++i) {
      input_size *= input_tensor.dim_size(i);
    }
    int input_dims = shape_tensor.dim_size(0);
    int cols = shape[input_dims - 1];
    int batch_size = 1;
    for (int i = 0; i < input_dims - 1; ++i) {
      batch_size *= shape[i];
    }
    OP_REQUIRES(context, (input_size == batch_size * cols),
                errors::InvalidArgument("input id is dense"));
    int embedding_dims = weight_tensor->dim_size(1);
    bool is_mean = (combiner_ == 1);

    Tensor* output_tensor = NULL;
    TensorShape output_shape({batch_size, embedding_dims});
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
    float *output = (float *)output_tensor->tensor_data().data();
    ktfop::EmbeddingParams params(input,
                                  batch_size,
                                  cols,
                                  weight,
                                  embedding_dims,
                                  is_mean);
    int result = ktfop::FusedEmbedding(params, output);
    OP_REQUIRES(context, (result == 0),
                errors::InvalidArgument("Invalid argument, error code: ", result));
  }

private:
  int combiner_;
  std::string node_name;
};

REGISTER_KERNEL_BUILDER(Name("KPFusedEmbedding").Device(DEVICE_CPU), KPFusedEmbeddingOp);

class KPFusedEmbeddingWithHashBucketOp : public OpKernel {
    public:
      explicit KPFusedEmbeddingWithHashBucketOp(OpKernelConstruction* context)
               : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("num_buckets", &num_buckets_));
        node_name = context->def().name();
      }

      void Compute(OpKernelContext* context) override {
        float *weight;
        const Tensor& input_tensor = context->input(0);
        const Tensor* weight_tensor = &context->input(1);
        
        if (weight_tensor->dtype() == DT_RESOURCE) {
          Var* variable;
          OP_REQUIRES_OK(context,
                        LookupResource(context, HandleFromInput(context, 1), 
                                        &variable));
          core::ScopedUnref s(variable);
          weight_tensor = variable->tensor();
          OP_REQUIRES(context, weight_tensor->dtype() == DT_FLOAT,
                      errors::InvalidArgument("Expect float weight in ",
                                          node_name));
        }
        
        auto input = input_tensor.flat<tstring>();
        weight = (float *)weight_tensor->tensor_data().data();
        int64_t batch = input_tensor.dim_size(0);
        int64_t embedding_dims = weight_tensor->dim_size(1);
        uintptr_t cstr_addresses[batch];
        size_t cstr_length[batch];
        for (int i = 0; i < batch; ++i) {
          cstr_addresses[i] = reinterpret_cast<uintptr_t>(input(i).c_str());
          cstr_length[i] = input(i).length();
        }
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, 
                       context->allocate_output(
                        0, TensorShape({batch, embedding_dims}), 
                        &output_tensor));
        float *output = (float *)output_tensor->tensor_data().data();
        ktfop::EmbeddingParamsWithHash params(cstr_addresses,
                                              cstr_length,
                                              batch,
                                              weight,
                                              num_buckets_,
                                              embedding_dims);
        int result = ktfop::FusedEmbeddingWithHashBucket(params, output);
        OP_REQUIRES(context, (result == 0),
                errors::InvalidArgument("Invalid argument, error code: ", result));
      }

    private:
        int64_t num_buckets_;
        std::string node_name;
};
REGISTER_KERNEL_BUILDER(Name("KPFusedEmbeddingWithHashBucket").Device(DEVICE_CPU), KPFusedEmbeddingWithHashBucketOp);
}  // namespace tensorflow
