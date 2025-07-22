#include "kdnn.hpp"
#include "kdnn_threadpool.h"

namespace tensorflow {

inline void kdnnGemm(OpKernelContext* ctx, const Tensor& a, const Tensor& b, Tensor* out,
                     bool trans_a_, bool trans_b) {
    int m = a.dim_size(0);
    int n = b.dim_size(1);
    int k = a.dim_size(1);
    const float *A = a.flat<float>().data();
    const float *B = b.flat<float>().data();
    float *C = out->flat<float>().data();
    // intra_op thread_pool
    thread::ThreadPool* thread_pool = 
        ctx->device()
        ->tensorflow_cpu_worker_threads()
        ->workers;
    kdnn::KDNNThreadPool eigen_tp(thread_pool);
    const KDNN::TensorInfo srcInfo = {{m, k}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
    const KDNN::TensorInfo weightsInfo = {{k, n}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
    const KDNN::TensorInfo dstInfo = {{m, n}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
    KDNN::Gemm gemm(srcInfo, weightsInfo, dstInfo, &eigen_tp);
    gemm.Run(A, B, C);
}

}// namespace tensorflow