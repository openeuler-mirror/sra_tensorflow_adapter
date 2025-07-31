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

struct NonZeroElement {
  int row;
  int col;
  float val;
};

static bool compareByRow(const NonZeroElement& a, const NonZeroElement& b) {
  return a.row < b.row;
}

template<typename Tindices>
void kdnnSparseMatmul(const std::size_t nnz,
                             const std::size_t rhs_right, const std::size_t lhs_right,
                             typename TTypes<float>::Matrix out,
                             typename TTypes<Tindices>::ConstMatrix a_indices, 
                             typename TTypes<float>::ConstVec a_values,
                             typename TTypes<float>::ConstMatrix b) {
    KDNN_INT idx[nnz];
    float val[nnz];
    std::vector<NonZeroElement> elements;
    for (size_t i = 0; i < nnz; ++i) {
        elements.emplace_back(NonZeroElement{a_indices(i, 0), a_indices(i, 1), a_values(i)});
    }
    std::sort(elements.begin(), elements.end(), compareByRow);
    for (size_t i = 0; i < nnz; ++i) {
        idx[i] = elements[i].col;
        val[i] = elements[i].val;
    }
    int m = out.dimension(0);
    KDNN_INT pntrb[m] = {0};
    KDNN_INT pntre[m] = {0};
    std::vector<int> row_counts(m, 0);
    for (const auto& t : elements) {
        row_counts[t.row]++;
    }
    int current_pos = 0;
    for (size_t i = 0; i < m; ++i) {
        pntrb[i] = current_pos;
        current_pos += row_counts[i];
        pntre[i] = current_pos;
    }
    VLOG(1) << "kdnnSparseMatmul, M: " << m << "  N:" << rhs_right << "  K:" << lhs_right << "  nnz:" << nnz;
    KDNN::SparseCsrmm(KDNN_SPARSE_OPERATION_NON_TRANSPOSE, m, rhs_right, lhs_right, 
                        1.0, "G00C", val, idx, pntrb, pntre, b.data(), rhs_right, 0.0, out.data(), rhs_right);
}

}// namespace tensorflow