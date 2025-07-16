#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;

template <typename Tidx>
class KPFusedSparseSegmentReduceOp : public OpKernel {
 public:
  explicit KPFusedSparseSegmentReduceOp(OpKernelConstruction* context)
      : OpKernel(context) {
    int combiner_mode;
    OP_REQUIRES_OK(context, context->GetAttr("combiner", &combiner_mode));
    OP_REQUIRES(context, combiner_mode == 0 || combiner_mode == 1,
                errors::InvalidArgument("combiner must be 0 or 1"));
    is_mean_ = (combiner_mode == 1);
  }

  void Compute(OpKernelContext* context) override {
    VLOG(0) << "Executing KPFusedSparseSegmentReduce operator";
    const Tensor& data = context->input(0);
    const Tensor& indices = context->input(1);
    const Tensor& slice_input = context->input(2);
    const Tensor& begin = context->input(3);

    // Log input shapes
    VLOG(0) << "Input data shape: " << data.shape().DebugString();
    VLOG(0) << "Input indices shape: " << indices.shape().DebugString();
    VLOG(0) << "Input slice_input shape: " << slice_input.shape().DebugString();
    VLOG(0) << "Input begin value: " << begin.SummarizeValue(10);

    int64_t num_indices = indices.dim_size(0);
    int64_t embedding_size = data.dim_size(1);
    int32 col = begin.flat<int32>().data()[1];

    auto data_mat = data.matrix<float>();
    auto indices_vec = indices.vec<Tidx>();
    auto slice_input_mat = slice_input.matrix<int64>();

    VLOG(0) << "Number of indices: " << num_indices;
    VLOG(0) << "Embedding size: " << embedding_size;
    VLOG(0) << "Column index from begin: " << col;

    // Calculate max segment_id
    int64 max_seg_id = 0;
    for (int32 i = 0; i < num_indices; ++i) {
      int64 seg_id = slice_input_mat(i, col);
      if (seg_id > max_seg_id) {
        max_seg_id = seg_id;
      }
    }
    const int64 batch_size = max_seg_id + 1;
    VLOG(0) << "Calculated batch size: " << batch_size;

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({batch_size, embedding_size}), &output));
    output->flat<float>().setZero();
    Tensor* slice_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({}), &slice_out));
    slice_out->scalar<int32>()() = batch_size;

    // KDNN::fused_compute(data, indices, slice_input, output, batch,
    // embedding_size, is_mean)

    auto output_mat = output->matrix<float>();

    if (is_mean_) {
      Tensor counts(DT_INT32, TensorShape({batch_size}));
      counts.flat<int32>().setZero();
      auto counts_vec = counts.flat<int32>();

      for (int64 i = 0; i < num_indices; ++i) {
        int32 seg_id = slice_input_mat(i, col);
        for (int64 j = 0; j < embedding_size; ++j) {
          output_mat(seg_id, j) += data_mat(indices_vec(i), j);
        }
        counts_vec(seg_id) += 1;
      }

      auto worker_threads = context->device()->tensorflow_cpu_worker_threads();
      const int64 cost_per_unit = 1000 * embedding_size;

      Shard(worker_threads->num_threads, worker_threads->workers, batch_size,
            cost_per_unit,
            [&output_mat, &counts_vec](int64 start_seg, int64 end_seg) {
              for (int32 seg = start_seg; seg < end_seg; ++seg) {
                const int32 count = counts_vec(seg);
                if (count > 0) {
                  for (int64 j = 0; j < output_mat.dimension(1); ++j) {
                    output_mat(seg, j) /= count;
                  }
                }
              }
            });
    } else {
      for (int64 i = 0; i < num_indices; ++i) {
        int32 seg_id = slice_input_mat(i, col);
        for (int64 j = 0; j < embedding_size; ++j) {
          output_mat(seg_id, j) += data_mat(indices_vec(i), j);
        }
      }
    }
  }

 private:
  bool is_mean_;
};

#define REGISTER_KERNEL(Tidx) \
    REGISTER_KERNEL_BUILDER(                                         \
        Name("KPFusedSparseSegmentReduce")                           \
            .Device(DEVICE_CPU)                                      \
            .TypeConstraint<Tidx>("Tidx"),                           \
            KPFusedSparseSegmentReduceOp<Tidx>);
REGISTER_KERNEL(int64)
REGISTER_KERNEL(int32)
#undef REGISTER_KERNEL