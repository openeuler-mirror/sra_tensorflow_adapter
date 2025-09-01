/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_KDNN_KDNN_ADAPTER_H_
#define THIRD_PARTY_KDNN_KDNN_ADAPTER_H_

#include "kdnn.hpp"
#include "kdnn_threadpool.h"
#include "tensorflow/core/util/matmul_bcast.h"

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
    kdnn::KDNNThreadPool kdnn_tp(thread_pool);
    const KDNN::TensorInfo srcInfo = {{m, k}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
    const KDNN::TensorInfo weightsInfo = {{k, n}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
    const KDNN::TensorInfo dstInfo = {{m, n}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
    KDNN::Gemm gemm(srcInfo, weightsInfo, dstInfo, &kdnn_tp);
    gemm.Run(A, B, C);
}

inline void kdnnParallelGemm(const OpKernelContext* ctx, const Tensor& a, const Tensor& b, Tensor* out,
                     const MatMulBCast& bcast, int start, int end) {
  const bool should_bcast = bcast.IsBroadcastingRequired();
  const auto& x_batch_indices = bcast.x_batch_indices();
  const auto& y_batch_indices = bcast.y_batch_indices();
  int m = a.dim_size(1);
  int n = b.dim_size(2);
  int k = a.dim_size(2);
  int stride_a = m * k;
  int stride_b = k * n;
  int stride_c = m * n;
  const float *A = a.flat<float>().data();
  const float *B = b.flat<float>().data();
  float *C = out->flat<float>().data();
  // intra_op thread_pool
  thread::ThreadPool* thread_pool = 
    ctx->device()
    ->tensorflow_cpu_worker_threads()
    ->workers;
  kdnn::KDNNThreadPool kdnn_tp(thread_pool);
  const KDNN::TensorInfo srcInfo = {{m, k}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
  const KDNN::TensorInfo weightsInfo = {{k, n}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
  const KDNN::TensorInfo dstInfo = {{m, n}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
  KDNN::Gemm gemm(srcInfo, weightsInfo, dstInfo, &kdnn_tp);
  for (int64_t i = start; i < end; ++i) {
    const int64_t x_batch_index = should_bcast ? x_batch_indices[i] : i;
    const int64_t y_batch_index = should_bcast ? y_batch_indices[i] : i;
    gemm.Run(A + x_batch_index * stride_a, B + y_batch_index * stride_b, C + i * stride_c); 
  }
}

inline void kdnnSeqGemm(const Tensor& a, const Tensor& b, Tensor* out,
                    const MatMulBCast& bcast, int start, int end) {
  const bool should_bcast = bcast.IsBroadcastingRequired();
  const auto& x_batch_indices = bcast.x_batch_indices();
  const auto& y_batch_indices = bcast.y_batch_indices();
  int m = a.dim_size(1);
  int n = b.dim_size(2);
  int k = a.dim_size(2);
  int stride_a = m * k;
  int stride_b = k * n;
  int stride_c = m * n;
  const float *A = a.flat<float>().data();
  const float *B = b.flat<float>().data();
  float *C = out->flat<float>().data();
  const KDNN::TensorInfo srcInfo = {{m, k}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
  const KDNN::TensorInfo weightsInfo = {{k, n}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
  const KDNN::TensorInfo dstInfo = {{m, n}, KDNN::Element::TypeT::F32, KDNN::Layout::AB};
  KDNN::Gemm gemm(srcInfo, weightsInfo, dstInfo);
  for (int64_t i = start; i < end; ++i) {
    const int64_t x_batch_index = should_bcast ? x_batch_indices[i] : i;
    const int64_t y_batch_index = should_bcast ? y_batch_indices[i] : i;
    gemm.Run(A + x_batch_index * stride_a, B + y_batch_index * stride_b, C + i * stride_c); 
  }
}

template<typename Tindices>
inline void kdnnSparseMatmulCSR(const std::size_t nnz,
                      const std::size_t rhs_right, const std::size_t lhs_right,
                      const int lhs_index_a, const int rhs_index_a,
                      typename TTypes<float>::Matrix out,
                      typename TTypes<Tindices>::ConstMatrix a_indices, 
                      typename TTypes<float>::ConstVec a_values,
                      typename TTypes<float>::ConstMatrix b) {
    std::vector<int> idx(nnz);
    int m = out.dimension(0);
    std::vector<int> pntrb(m);
    std::vector<int> pntre(m);
    std::vector<int> row_counts(m);
    for (size_t i = 0; i < nnz; ++i) {
        idx[i] = a_indices(i, rhs_index_a);
        ++row_counts[a_indices(i, lhs_index_a)];
    }
    
    int current_pos = 0;
    for (size_t i = 0; i < m; ++i) {
        pntrb[i] = current_pos;
        current_pos += row_counts[i];
        pntre[i] = current_pos;
    }
    const KDNN::CsrSparseTensorInfo aInfo = {{m, lhs_right},
        KDNN::Element::TypeT::F32, KDNN::Layout::AB, pntrb, pntre, idx, nnz};
    const KDNN::TensorInfo bInfo = {{lhs_right, rhs_right},
        KDNN::Element::TypeT::F32, KDNN::Layout::AB};
    const KDNN::TensorInfo dstInfo = {{m, rhs_right},
        KDNN::Element::TypeT::F32, KDNN::Layout::AB};
    KDNN::SparseGemm sparse_csr(aInfo, bInfo, dstInfo);
    sparse_csr.Run(a_values.data(), b.data(), out.data());
}

template<typename Tindices>
void kdnnSparseMatmul(const std::size_t nnz,
                      const std::size_t rhs_right, const std::size_t lhs_right,
                      const int lhs_index_a, const int rhs_index_a,
                      typename TTypes<float>::Matrix out,
                      typename TTypes<Tindices>::ConstMatrix a_indices, 
                      typename TTypes<float>::ConstVec a_values,
                      typename TTypes<float>::ConstMatrix b) {
    static const std::size_t kNumCSR = 720;
    int m = out.dimension(0);
    VLOG(1) << "kdnnSparseMatmul, M: " << m << "  N:" << rhs_right << "  K:" << lhs_right << "  nnz:" << nnz;
    if ((m > kint32max) || (rhs_right > kint32max) || (lhs_right > kint32max)) {
        LOG(WARNING) << "too large m/n/k in KDNN sparse matmul, max allowed is " << kint32max;
        return;
    }
    kdnnSparseMatmulCSR<Tindices>(nnz, rhs_right, lhs_right, lhs_index_a, rhs_index_a, out, a_indices, a_values, b);
}

}// namespace tensorflow

#endif  // THIRD_PARTY_KDNN_KDNN_ADAPTER_H_