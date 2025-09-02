/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "absl/algorithm/container.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

template <typename T>
class MatMulOpTest : public OpsTestBase {
 protected:  
  using MatMulGraphRunner = 
      std::function<void(const Tensor& lhs_data, const Tensor& rhs_data,
                         Tensor* out)>;

    void RunAndFetch(const tensorflow::Scope& root, const string& fetch,
                   Tensor* output, bool allow_gpu_device,
                   const NodeDef* fetch_node = nullptr) {
    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    if (fetch_node) {
      *graph.add_node() = *fetch_node;
    }

    // We really want to make sure that graph executed exactly as we passed it
    // to the session, so we disable various optimizations.
    tensorflow::SessionOptions session_options;

    // Disable common runtime constant folding.
    session_options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_opt_level(OptimizerOptions::L0);

    // Disable Grappler optimizations for tests.
    tensorflow::RewriterConfig* cfg =
        session_options.config.mutable_graph_options()
            ->mutable_rewrite_options();
    cfg->set_constant_folding(tensorflow::RewriterConfig::OFF);
    cfg->set_layout_optimizer(tensorflow::RewriterConfig::OFF);
    cfg->set_remapping(tensorflow::RewriterConfig::OFF);

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(session_options));

    std::vector<DeviceAttributes> available_devices;
    TF_ASSERT_OK(session->ListDevices(&available_devices))
        << "Failed to get available session devices";

    // Check if session has an available GPU device.
    const bool has_gpu_device =
        absl::c_any_of(available_devices, [](const DeviceAttributes& device) {
          return device.device_type() == DEVICE_GPU;
        });

    // If fused computation implemented only for CPU, in this test we don't want
    // to compare GPU vs CPU numbers, so place all nodes on CPU in this case.
    const bool place_all_on_gpu = allow_gpu_device && has_gpu_device;

    const string device = place_all_on_gpu ? "/device:GPU:0" : "/device:CPU:0";
    for (NodeDef& mutable_node : *graph.mutable_node()) {
      mutable_node.set_device(device);
    }

    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> unfused_tensors;
    TF_ASSERT_OK(session->Run({}, {fetch}, {}, &unfused_tensors));

    *output = unfused_tensors[0];
  }

  void RunRefMatMul(const Tensor& a, const Tensor& b,
                bool transpose_a, bool transpose_b, Tensor* out,
                bool allow_gpu_device = false) {
    auto lhs = a.flat<T>().data();
    auto rhs = b.flat<T>().data();

    auto a_dim = a.shape().dim_sizes();
    auto b_dim = b.shape().dim_sizes();

    int m = a_dim[0];
    int k = a_dim[1];
    int n = b_dim[1];

    TensorShape out_shape({m, n});
    *out = Tensor(DataTypeToEnum<T>::v(), out_shape);
    auto output = out->flat<T>().data();

    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        T sum = T(0);
        for (int p = 0; p < k; ++p) {
          T a_val = lhs[i * k + p];
          T b_val = rhs[p * n + j];
          sum += a_val * b_val;
        }
        output[i * n + j] = sum;
      }
    }
  }

  void RunKMatMul(const Tensor& lhs_data, const Tensor& rhs_data,
                bool transpose_a, bool transpose_b, Tensor* output,
                bool allow_gpu_device = false) {
    Scope root = tensorflow::Scope::NewRootScope();

    ops::MatMul kmatmul = ops::MatMul(
      root.WithOpName("kmatmul"),
      ops::Const(root.WithOpName("lhs"), Input::Initializer(lhs_data)),
      ops::Const(root.WithOpName("rhs"), Input::Initializer(rhs_data)),
      ops::MatMul::Attrs().TransposeA(transpose_a).TransposeB(transpose_b));
    
    RunAndFetch(root, "kmatmul", output, allow_gpu_device);
  }

  void VerifyMatMulTensorsNear(
      const Tensor& lhs,
      const Tensor& rhs,
      const MatMulGraphRunner& run_reference,
      const MatMulGraphRunner& run_kdnn) {

    Tensor matmul;
    Tensor kmatmul;

    run_reference(lhs, rhs, &matmul);
    run_kdnn(lhs, rhs, &kmatmul);

    ASSERT_EQ(matmul.dtype(), kmatmul.dtype());
    ASSERT_EQ(matmul.shape(), kmatmul.shape());

    // 数值对比（允许浮点误差）
    test::ExpectClose(matmul, kmatmul, /*atol=*/1e-5);
  }

  void VerifyMatMul(int m, int k, int n, bool transpose_a, bool transpose_b) {
    DataType dtype = DataTypeToEnum<T>::v();
    Tensor lhs(dtype, {m, k});
    lhs.flat<T>() = lhs.flat<T>().setRandom();
    Tensor rhs(dtype, {k, n});
    rhs.flat<T>() = rhs.flat<T>().setRandom();

    const MatMulGraphRunner run_reference =
        [&](const Tensor& a, const Tensor& b,
            Tensor* out) {
          RunRefMatMul(a, b, false, false, out, false);
        };

    const MatMulGraphRunner run_kdnn =
        [&](const Tensor& a, const Tensor& b,
            Tensor* out) {
          RunKMatMul(a, b, false, false, out, false);
        };

    VerifyMatMulTensorsNear(lhs, rhs, run_reference, run_kdnn);
  }

  void VerifyMatMulWithInputs(
      int m, int k, int n,
      const std::vector<T>& A_data,
      const std::vector<T>& B_data,
      bool transpose_a = false,
      bool transpose_b = false) {
    ASSERT_EQ(A_data.size(), static_cast<size_t>(m * k));
    ASSERT_EQ(B_data.size(), static_cast<size_t>(k * n));

    DataType dtype = DataTypeToEnum<T>::v();

    Tensor lhs(dtype, {m, k});
    Tensor rhs(dtype, {k, n});
    std::copy(A_data.begin(), A_data.end(), lhs.flat<T>().data());
    std::copy(B_data.begin(), B_data.end(), rhs.flat<T>().data());

    const MatMulGraphRunner run_reference =
        [&](const Tensor& a, const Tensor& b, Tensor* out) {
          RunRefMatMul(a, b, transpose_a, transpose_b, out, false);
        };

    const MatMulGraphRunner run_kdnn =
        [&](const Tensor& a, const Tensor& b, Tensor* out) {
          RunKMatMul(a, b, transpose_a, transpose_b, out, false);
        };

    VerifyMatMulTensorsNear(lhs, rhs, run_reference, run_kdnn);
  }
};

TYPED_TEST_SUITE_P(MatMulOpTest);

template <typename T>
class FusedMatMulOpTest : public OpsTestBase {
 protected:
  using BiasAddGraphRunner =
      std::function<void(const Tensor& lhs_data, const Tensor& rhs_data,
                         const Tensor& bias_data, Tensor* out)>;

  // Runs a Tensorflow graph defined by the root scope, and fetches the result
  // of 'fetch' node into the output Tensor. Optional `fetch_node` parameter
  // allows to define a fetch node directly using a NodeDef for the ops that are
  // not supported by the C++ Api.
  void RunAndFetch(const tensorflow::Scope& root, const string& fetch,
                   Tensor* output, bool allow_gpu_device,
                   const NodeDef* fetch_node = nullptr) {
    tensorflow::GraphDef graph;
    TF_ASSERT_OK(root.ToGraphDef(&graph));

    if (fetch_node) {
      *graph.add_node() = *fetch_node;
    }

    // We really want to make sure that graph executed exactly as we passed it
    // to the session, so we disable various optimizations.
    tensorflow::SessionOptions session_options;

    // Disable common runtime constant folding.
    session_options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_opt_level(OptimizerOptions::L0);

    // Disable Grappler optimizations for tests.
    tensorflow::RewriterConfig* cfg =
        session_options.config.mutable_graph_options()
            ->mutable_rewrite_options();
    cfg->set_constant_folding(tensorflow::RewriterConfig::OFF);
    cfg->set_layout_optimizer(tensorflow::RewriterConfig::OFF);
    cfg->set_remapping(tensorflow::RewriterConfig::OFF);

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(session_options));

    std::vector<DeviceAttributes> available_devices;
    TF_ASSERT_OK(session->ListDevices(&available_devices))
        << "Failed to get available session devices";

    // Check if session has an available GPU device.
    const bool has_gpu_device =
        absl::c_any_of(available_devices, [](const DeviceAttributes& device) {
          return device.device_type() == DEVICE_GPU;
        });

    // If fused computation implemented only for CPU, in this test we don't want
    // to compare GPU vs CPU numbers, so place all nodes on CPU in this case.
    const bool place_all_on_gpu = allow_gpu_device && has_gpu_device;

    const string device = place_all_on_gpu ? "/device:GPU:0" : "/device:CPU:0";
    for (NodeDef& mutable_node : *graph.mutable_node()) {
      mutable_node.set_device(device);
    }

    TF_ASSERT_OK(session->Create(graph));

    std::vector<Tensor> unfused_tensors;
    TF_ASSERT_OK(session->Run({}, {fetch}, {}, &unfused_tensors));

    *output = unfused_tensors[0];
  }

  void RunMatMulWithBias(const Tensor& lhs_data, const Tensor& rhs_data,
                         const Tensor& bias_data, bool transpose_a,
                         bool transpose_b, Tensor* output,
                         bool allow_gpu_device = false) {
    Scope root = tensorflow::Scope::NewRootScope();

    ops::MatMul matmul = ops::MatMul(
        root.WithOpName("matmul"),
        ops::Const(root.WithOpName("lhs"), Input::Initializer(lhs_data)),
        ops::Const(root.WithOpName("rhs"), Input::Initializer(rhs_data)),
        ops::MatMul::Attrs().TransposeA(transpose_a).TransposeB(transpose_b));

    ops::BiasAdd with_bias = ops::BiasAdd(
        root.WithOpName("with_bias"), matmul,
        ops::Const(root.WithOpName("bias"), Input::Initializer(bias_data)));

    RunAndFetch(root, "with_bias", output, allow_gpu_device);
  }

  void RunMatMulWithBiasAndActivation(
      const Tensor& lhs_data, const Tensor& rhs_data, const Tensor& bias_data,
      bool transpose_a, bool transpose_b, const string& activation_type,
      Tensor* output, bool allow_gpu_device = false) {
    Scope root = tensorflow::Scope::NewRootScope();

    ops::MatMul matmul = ops::MatMul(
        root.WithOpName("matmul"),
        ops::Const(root.WithOpName("lhs"), Input::Initializer(lhs_data)),
        ops::Const(root.WithOpName("rhs"), Input::Initializer(rhs_data)),
        ops::MatMul::Attrs().TransposeA(transpose_a).TransposeB(transpose_b));

    ops::BiasAdd with_bias = ops::BiasAdd(
        root.WithOpName("with_bias"), matmul,
        ops::Const(root.WithOpName("bias"), Input::Initializer(bias_data)));

    if (activation_type == "Relu") {
      ops::Relu(root.WithOpName("with_activation"), with_bias);
    } else if (activation_type == "Relu6") {
      ops::Relu6(root.WithOpName("with_activation"), with_bias);
    } else if (activation_type == "Elu") {
      ops::Elu(root.WithOpName("with_activation"), with_bias);
    } else {
      ops::Identity(root.WithOpName("with_activation"), with_bias);
    }

    RunAndFetch(root, "with_activation", output, allow_gpu_device);
  }

  void RunFusedMatMulOp(const Tensor& lhs_data, const Tensor& rhs_data,
                        const std::vector<Tensor>& args_data,
                        const std::vector<string>& fused_ops, bool transpose_a,
                        bool transpose_b, Tensor* output,
                        bool allow_gpu_device = false) {
    Scope root = tensorflow::Scope::NewRootScope();

    DataType dtype = DataTypeToEnum<T>::v();
    int num_args = static_cast<int>(args_data.size());

    Output lhs =
        ops::Const(root.WithOpName("lhs"), Input::Initializer(lhs_data));
    Output rhs =
        ops::Const(root.WithOpName("rhs"), Input::Initializer(rhs_data));

    std::vector<NodeDefBuilder::NodeOut> args;
    for (int i = 0; i < num_args; ++i) {
      Output arg = ops::Const(root.WithOpName(absl::StrCat("arg", i)),
                              Input::Initializer(args_data[i]));
      args.emplace_back(arg.name(), 0, dtype);
    }

    NodeDef fused_matmul;
    TF_EXPECT_OK(NodeDefBuilder("fused_matmul", "_FusedMatMul")
                     .Input({lhs.name(), 0, dtype})
                     .Input({rhs.name(), 0, dtype})
                     .Input(args)
                     .Attr("num_args", num_args)
                     .Attr("T", dtype)
                     .Attr("fused_ops", fused_ops)
                     .Attr("transpose_a", transpose_a)
                     .Attr("transpose_b", transpose_b)
                     .Finalize(&fused_matmul));

    RunAndFetch(root, fused_matmul.name(), output, allow_gpu_device,
                &fused_matmul);
  }

  void VerifyBiasAddTensorsNear(int m, int k, int n,
                                const BiasAddGraphRunner& run_default,
                                const BiasAddGraphRunner& run_fused) {
    DataType dtype = DataTypeToEnum<T>::v();

    Tensor lhs(dtype, {m, k});
    lhs.flat<T>() = lhs.flat<T>().setRandom();

    // Add some negative values to filter to properly test Relu.
    Tensor rhs(dtype, {k, n});
    rhs.flat<T>() = rhs.flat<T>().setRandom();
    rhs.flat<T>() -= rhs.flat<T>().constant(static_cast<T>(0.5f));

    // Bias added to the inner dimension.
    const int bias_size = n;
    Tensor bias(dtype, {bias_size});
    bias.flat<T>() = bias.flat<T>().setRandom();
    bias.flat<T>() += bias.flat<T>().constant(static_cast<T>(0.5f));

    Tensor matmul;
    Tensor fused_matmul;

    run_default(lhs, rhs, bias, &matmul);
    run_fused(lhs, rhs, bias, &fused_matmul);

    ASSERT_EQ(matmul.dtype(), fused_matmul.dtype());
    ASSERT_EQ(matmul.shape(), fused_matmul.shape());

    test::ExpectClose(matmul, fused_matmul, /*atol=*/1e-5);
  }

  // Verifies that computing MatMul+BiasAdd in a graph is identical to
  // FusedMatMul.
  void VerifyMatMulWithBias(int m, int k, int n, bool transpose_a,
                            bool transpose_b) {
    const BiasAddGraphRunner run_default =
        [&](const Tensor& input_data, const Tensor& filter_data,
            const Tensor& bias_data, Tensor* out) {
          RunMatMulWithBias(input_data, filter_data, bias_data, transpose_a,
                            transpose_b, out);
        };

    const BiasAddGraphRunner run_fused =
        [&](const Tensor& input_data, const Tensor& filter_data,
            const Tensor& bias_data, Tensor* out) {
          RunFusedMatMulOp(input_data, filter_data, {bias_data}, {"BiasAdd"},
                           transpose_a, transpose_b, out);
        };

    VerifyBiasAddTensorsNear(m, k, n, run_default, run_fused);
  }

  // Verifies that computing MatMul+BiasAdd+{Activation} in a graph is identical
  // to FusedMatMul.
  void VerifyConv2DWithBiasAndActivation(int m, int k, int n, bool transpose_a,
                                         bool transpose_b,
                                         const string& activation) {
    const BiasAddGraphRunner run_default = [&](const Tensor& input_data,
                                               const Tensor& filter_data,
                                               const Tensor& bias_data,
                                               Tensor* out) {
      RunMatMulWithBiasAndActivation(input_data, filter_data, bias_data,
                                     transpose_a, transpose_b, activation, out);
    };

    const BiasAddGraphRunner run_fused = [&](const Tensor& input_data,
                                             const Tensor& filter_data,
                                             const Tensor& bias_data,
                                             Tensor* out) {
      RunFusedMatMulOp(input_data, filter_data, {bias_data},
                       {"BiasAdd", activation}, transpose_a, transpose_b, out);
    };

    VerifyBiasAddTensorsNear(m, k, n, run_default, run_fused);
  }
};

// MatMul with BatchNorm can be tested only with `T=float`, because default
// `FusedBatchNorm` kernel supports only floats for scale, mean and variance.

template <typename T>
class FusedMatMulWithBiasOpTest : public FusedMatMulOpTest<T> {};

TYPED_TEST_SUITE_P(FusedMatMulWithBiasOpTest);

// -------------------------------------------------------------------------- //
// MatMul + BiasAdd + {Activation}                                            //
// -------------------------------------------------------------------------- //

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul256x256x256) {
  this->VerifyMatMulWithBias(256, 256, 256, false, false);
  this->VerifyMatMulWithBias(256, 256, 256, true, false);
  this->VerifyMatMulWithBias(256, 256, 256, false, true);
  this->VerifyMatMulWithBias(256, 256, 256, true, true);
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul1x256x256) {
  this->VerifyMatMulWithBias(1, 256, 256, false, false);
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul256x256x1) {
  this->VerifyMatMulWithBias(256, 256, 1, false, false);
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul1x256x1) {
  this->VerifyMatMulWithBias(1, 256, 1, false, false);
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul256x256x256WithActivation) {
  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    this->VerifyConv2DWithBiasAndActivation(256, 256, 256, false, false,
                                            activation);
    this->VerifyConv2DWithBiasAndActivation(256, 256, 256, true, false,
                                            activation);
    this->VerifyConv2DWithBiasAndActivation(256, 256, 256, false, true,
                                            activation);
    this->VerifyConv2DWithBiasAndActivation(256, 256, 256, true, true,
                                            activation);
  }
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul1x256x256WithActivation) {
  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    this->VerifyConv2DWithBiasAndActivation(1, 256, 256, false, false,
                                            activation);
  }
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul256x256x1WithActivation) {
  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    this->VerifyConv2DWithBiasAndActivation(256, 256, 1, false, false,
                                            activation);
  }
}

TYPED_TEST_P(FusedMatMulWithBiasOpTest, MatMul1x256x1WithActivation) {
  for (const string& activation : {"Relu", "Relu6", "Elu"}) {
    this->VerifyConv2DWithBiasAndActivation(1, 256, 1, false, false,
                                            activation);
  }
}

// -------------------------------------------------------------------------- //
// MatMul Base Random Test                                                    //
// -------------------------------------------------------------------------- //


// 基础维度
TYPED_TEST_P(MatMulOpTest, MatMul_256x256x256) {
  this->VerifyMatMul(256, 256, 256, false, false);
}

TYPED_TEST_P(MatMulOpTest, MatMul_1x256x256) {
  this->VerifyMatMul(1, 256, 256, false, false);
}

TYPED_TEST_P(MatMulOpTest, MatMul_256x256x1) {
  this->VerifyMatMul(256, 256, 1, false, false);
}

TYPED_TEST_P(MatMulOpTest, MatMul_1x256x1) {
  this->VerifyMatMul(1, 256, 1, false, false);
}

// -------------------------------------------------------------------------- //
// MatMul Extended Random Test                                                //
// Cover: jdtest                                                              //
// -------------------------------------------------------------------------- //

// === 中等规模 & KDNN 常见场景 ===
TYPED_TEST_P(MatMulOpTest, MatMul_5530x104x32) {
  this->VerifyMatMul(5530, 104, 32, false, false);
}
TYPED_TEST_P(MatMulOpTest, MatMul_5530x116x32) {
  this->VerifyMatMul(5530, 116, 32, false, false);
}
TYPED_TEST_P(MatMulOpTest, MatMul_5530x32x16) {
  this->VerifyMatMul(5530, 32, 16, false, false);
}
TYPED_TEST_P(MatMulOpTest, MatMul_5530x16x1) {
  this->VerifyMatMul(5530, 16, 1, false, false);
}
TYPED_TEST_P(MatMulOpTest, MatMul_7000x104x32) {
  this->VerifyMatMul(7000, 104, 32, false, false);
}
TYPED_TEST_P(MatMulOpTest, MatMul_7000x116x32) {
  this->VerifyMatMul(7000, 116, 32, false, false);
}
TYPED_TEST_P(MatMulOpTest, MatMul_7000x32x16) {
  this->VerifyMatMul(7000, 32, 16, false, false);
}
TYPED_TEST_P(MatMulOpTest, MatMul_7000x16x1) {
  this->VerifyMatMul(7000, 16, 1, false, false);
}

// 极小维度测试：0 维度（空矩阵）
TYPED_TEST_P(MatMulOpTest, MatMul_0x256x256) {
  this->VerifyMatMul(0, 256, 256, false, false);
}

TYPED_TEST_P(MatMulOpTest, MatMul_256x0x256) {
  this->VerifyMatMul(256, 0, 256, false, false);
}

TYPED_TEST_P(MatMulOpTest, MatMul_256x256x0) {
  this->VerifyMatMul(256, 256, 0, false, false);
}

TYPED_TEST_P(MatMulOpTest, MatMul_0x0x0) {
  this->VerifyMatMul(0, 0, 0, false, false);
}

TYPED_TEST_P(MatMulOpTest, MatMul_0x0x1) {
  this->VerifyMatMul(0, 0, 1, false, false);
}

TYPED_TEST_P(MatMulOpTest, MatMul_0x1x0) {
  this->VerifyMatMul(0, 1, 0, false, false);
}

TYPED_TEST_P(MatMulOpTest, MatMul_1x0x0) {
  this->VerifyMatMul(1, 0, 0, false, false);
}

// 非 2 的幂次维度（内存不对齐）
TYPED_TEST_P(MatMulOpTest, MatMul_257x257x257) {
  this->VerifyMatMul(257, 257, 257, false, false);
}

TYPED_TEST_P(MatMulOpTest, MatMul_250x240x230) {
  this->VerifyMatMul(250, 240, 230, false, false);
}

TYPED_TEST_P(MatMulOpTest, MatMul_123x456x789) {
  this->VerifyMatMul(123, 456, 789, false, false);
}

// 大 k 值（高计算密度）
TYPED_TEST_P(MatMulOpTest, MatMul_64x8192x64) {
  this->VerifyMatMul(64, 8192, 64, false, false);
}

// 小 k 值
TYPED_TEST_P(MatMulOpTest, MatMul_256x1x256) {
  this->VerifyMatMul(256, 1, 256, false, false);
}

// 大 k：如 Embedding 后接 FFN
TYPED_TEST_P(MatMulOpTest, MatMul_64x4096x512) {
  this->VerifyMatMul(64, 4096, 512, false, false);
}

// 小 m, 大 n：如分类头
TYPED_TEST_P(MatMulOpTest, MatMul_1x512x1000) {
  this->VerifyMatMul(1, 512, 1000, false, false);
}

// 大 m, 小 n：如 Batch 大但输出小
TYPED_TEST_P(MatMulOpTest, MatMul_1024x256x1) {
  this->VerifyMatMul(1024, 256, 1, false, false);
}

// 超小维度组合（广播/Kernel 选择错误）
TYPED_TEST_P(MatMulOpTest, MatMul_1x1x1) {
  this->VerifyMatMul(1, 1, 1, false, false);
}

TYPED_TEST_P(MatMulOpTest, MatMul_1x1x64) {
  this->VerifyMatMul(1, 1, 64, false, false);
}

TYPED_TEST_P(MatMulOpTest, MatMul_64x1x1) {
  this->VerifyMatMul(64, 1, 1, false, false);
}

// -------------------------------------------------------------------------- //
// MatMul Base Value Test                                                     //
// -------------------------------------------------------------------------- //

// 零值矩阵
TYPED_TEST_P(MatMulOpTest, MatMul_ZeroMatrix_A) {
  int m = 3, k = 4, n = 5;
  std::vector<TypeParam> A(m * k, TypeParam(0));  // 全零矩阵
  std::vector<TypeParam> B(k * n);
  std::generate(B.begin(), B.end(), []() { 
    return static_cast<TypeParam>(rand() % 10 - 5); 
  });

  this->VerifyMatMulWithInputs(m, k, n, A, B);
}

TYPED_TEST_P(MatMulOpTest, MatMul_ZeroMatrix_B) {
  int m = 3, k = 4, n = 5;
  std::vector<TypeParam> A(m * k);
  std::vector<TypeParam> B(k * n, TypeParam(0));
  std::generate(A.begin(), A.end(), []() { 
    return static_cast<TypeParam>(rand() % 10 - 5); 
  });

  this->VerifyMatMulWithInputs(m, k, n, A, B);
}

TYPED_TEST_P(MatMulOpTest, MatMul_ZeroMatrix_AB) {
  int m = 3, k = 4, n = 5;
  std::vector<TypeParam> A(m * k, TypeParam(0));
  std::vector<TypeParam> B(k * n, TypeParam(0));

  this->VerifyMatMulWithInputs(m, k, n, A, B);
}

// 单位矩阵
TYPED_TEST_P(MatMulOpTest, MatMul_IdentityMatrix_A) {
  int n = 9;  // 使用 9x9 单位阵
  int m = n, k = n;
  std::vector<TypeParam> A(m * k, TypeParam(0));
  for (int i = 0; i < n; ++i) {
    A[i * k + i] = TypeParam(1);  // 对角线为 1
  }

  std::vector<TypeParam> B(k * n);
  std::generate(B.begin(), B.end(), []() { 
    return static_cast<TypeParam>(rand() % 10 - 5); 
  });

  this->VerifyMatMulWithInputs(m, k, n, A, B);
}

TYPED_TEST_P(MatMulOpTest, MatMul_IdentityMatrix_B) {
  int n = 9;  // 使用 9x9 单位阵
  int m = n, k = n;
  std::vector<TypeParam> A(m * k);
  std::generate(A.begin(), A.end(), []() { 
    return static_cast<TypeParam>(rand() % 10 - 5); 
  });

  std::vector<TypeParam> B(k * n, TypeParam(0));
  for (int i = 0; i < n; ++i) {
    B[i * n + i] = TypeParam(1);  // 对角线为 1
  }

  this->VerifyMatMulWithInputs(m, k, n, A, B);
}

TYPED_TEST_P(MatMulOpTest, MatMul_IdentityMatrix_AB) {
  int n = 9;  // 使用 9x9 单位阵
  int m = n, k = n;
  std::vector<TypeParam> A(m * k, TypeParam(0));
  for (int i = 0; i < n; ++i) {
    A[i * k + i] = TypeParam(1);  // 对角线为 1
  }

  std::vector<TypeParam> B = A;

  this->VerifyMatMulWithInputs(m, k, n, A, B);
}

// 浮点特殊值
// | A = [[+inf]] | B =  | Expected = [[+inf]] |
TYPED_TEST_P(MatMulOpTest, MatMul_Inf_Positive) {
  std::vector<TypeParam> A = {std::numeric_limits<TypeParam>::infinity()};
  std::vector<TypeParam> B = {TypeParam(2)};

  this->VerifyMatMulWithInputs(1, 1, 1, A, B);
}

// | A = [[-inf]] | B =  | Expected = [[-inf]] |
TYPED_TEST_P(MatMulOpTest, MatMul_Inf_Negative) {
  std::vector<TypeParam> A = {-std::numeric_limits<TypeParam>::infinity()};
  std::vector<TypeParam> B = {TypeParam(2)};

  this->VerifyMatMulWithInputs(1, 1, 1, A, B);
}

// NaN 传播
// | A = [[nan]] | B = [[b]] | Expected = [[nan]] |
TYPED_TEST_P(MatMulOpTest, MatMul_NaN) {
  int m = 2, k = 2, n = 2;
  std::vector<TypeParam> A = {1.0, 2.0,
                              3.0, std::numeric_limits<TypeParam>::quiet_NaN()};

  std::vector<TypeParam> B = {1.0, 1.0,
                              1.0, 1.0};
  Tensor lhs(DataTypeToEnum<TypeParam>::v(), {m, k});
  Tensor rhs(DataTypeToEnum<TypeParam>::v(), {k, n});
  std::copy(A.begin(), A.end(), lhs.flat<TypeParam>().data());
  std::copy(B.begin(), B.end(), rhs.flat<TypeParam>().data());

  Tensor output_ref, output_kdnn;

  this->RunRefMatMul(lhs, rhs, false, false, &output_ref);
  this->RunKMatMul(lhs, rhs, false, false, &output_kdnn, false);
  auto ref_flat = output_ref.flat<TypeParam>();
  auto kdnn_flat = output_kdnn.flat<TypeParam>();
  int size = ref_flat.size();

  for (int i = 0; i < size; ++i) {
    TypeParam x = ref_flat(i);
    TypeParam y = kdnn_flat(i);

    bool both_nan = std::isnan(x) && std::isnan(y);
    bool both_inf = std::isinf(x) && std::isinf(y) && (std::signbit(x) == std::signbit(y));

    TypeParam atol = 1e-5f;
    TypeParam rtol = 1e-5f;
    TypeParam diff = std::abs(x - y);
    TypeParam threshold = atol + rtol * std::abs(y);
    bool normal_close = !std::isnan(x) && !std::isnan(y) && 
                        diff <= threshold;  // 自定义 atol/rtol

    EXPECT_TRUE(both_nan || both_inf || normal_close)
        << "Mismatch at index " << i << ": ref=" << x << ", kdnn=" << y;
  }
}

// 浮点数极限值
// | A = max | B = 1 | Expected = max |
TYPED_TEST_P(MatMulOpTest, MatMul_FloatMax_Times_One) {
  auto max_val = std::numeric_limits<TypeParam>::max();
  std::vector<TypeParam> A = {max_val};
  std::vector<TypeParam> B = {TypeParam(1)};

  this->VerifyMatMulWithInputs(1, 1, 1, A, B);
}

// | A = min | B = 1 | Expected = min |
TYPED_TEST_P(MatMulOpTest, MatMul_FloatMin_Times_One) {
  auto min_val = std::numeric_limits<TypeParam>::lowest();
  std::vector<TypeParam> A = {min_val};
  std::vector<TypeParam> B = {TypeParam(1)};

  this->VerifyMatMulWithInputs(1, 1, 1, A, B);
}

// (max-1) × 1 = max-1
TYPED_TEST_P(MatMulOpTest, MatMul_FloatMax_Minus_One_Times_One) {
  auto max_val = std::numeric_limits<TypeParam>::max();
  if (std::isfinite(max_val)) {
    std::vector<TypeParam> A = {max_val - TypeParam(1)};
    std::vector<TypeParam> B = {TypeParam(1)};
    this->VerifyMatMulWithInputs(1, 1, 1, A, B);
  }
}

// (min+1) × 1 = min+1
TYPED_TEST_P(MatMulOpTest, MatMul_FloatMin_Plus_One_Times_One) {
  auto min_val = std::numeric_limits<TypeParam>::lowest();
  if (std::isfinite(min_val)) {
    std::vector<TypeParam> A = {min_val + TypeParam(1)};
    std::vector<TypeParam> B = {TypeParam(1)};
    this->VerifyMatMulWithInputs(1, 1, 1, A, B);
  }
}

// (max/2) × 2 → 应溢出为 inf
TYPED_TEST_P(MatMulOpTest, MatMul_FloatMax_Half_Times_Two) {
  auto max_val = std::numeric_limits<TypeParam>::max();
  TypeParam half_max = max_val / TypeParam(2);
  std::vector<TypeParam> A = {half_max};
  std::vector<TypeParam> B = {TypeParam(2)};

  this->VerifyMatMulWithInputs(1, 1, 1, A, B);
}

// (min/2) × 2 → 应溢出为 -inf
TYPED_TEST_P(MatMulOpTest, MatMul_FloatMin_Half_Times_Two) {
  auto min_val = std::numeric_limits<TypeParam>::lowest();
  TypeParam half_min = min_val / TypeParam(2);
  std::vector<TypeParam> A = {half_min};
  std::vector<TypeParam> B = {TypeParam(2)};

  this->VerifyMatMulWithInputs(1, 1, 1, A, B);
}

// (max-1)/2 × 2 = max-1（不溢出）
TYPED_TEST_P(MatMulOpTest, MatMul_FloatMax_Minus_One_Half_Times_Two) {
  auto max_val = std::numeric_limits<TypeParam>::max();
  TypeParam val = (max_val - TypeParam(1)) / TypeParam(2);
  std::vector<TypeParam> A = {val};
  std::vector<TypeParam> B = {TypeParam(2)};

  this->VerifyMatMulWithInputs(1, 1, 1, A, B);
}

// (min+1)/2 × 2 = min+1（不溢出）
TYPED_TEST_P(MatMulOpTest, MatMul_FloatMin_Plus_One_Half_Times_Two) {
  auto min_val = std::numeric_limits<TypeParam>::lowest();
  TypeParam val = (min_val + TypeParam(1)) / TypeParam(2);
  std::vector<TypeParam> A = {val};
  std::vector<TypeParam> B = {TypeParam(2)};

  this->VerifyMatMulWithInputs(1, 1, 1, A, B);
}

// 精度敏感测试
TYPED_TEST_P(MatMulOpTest, MatMul_PrecisionSensitive) {
  // 使用小数值
  const TypeParam kSmall = TypeParam(0.01);

  int m = 2, k = 2, n = 2;
  std::vector<TypeParam> A = {kSmall, kSmall,
                              kSmall, kSmall};
  std::vector<TypeParam> B = {kSmall, kSmall,
                              kSmall, kSmall};

  // 期望结果：C[i][j] = 2 * (0.01 * 0.01) = 2 * 0.0001 = 0.0002
  const TypeParam kExpected = TypeParam(2) * kSmall * kSmall;  // 0.0002

  Tensor lhs(DataTypeToEnum<TypeParam>::v(), {m, k});
  Tensor rhs(DataTypeToEnum<TypeParam>::v(), {k, n});
  std::copy(A.begin(), A.end(), lhs.flat<TypeParam>().data());
  std::copy(B.begin(), B.end(), rhs.flat<TypeParam>().data());

  Tensor output;
  this->RunKMatMul(lhs, rhs, false, false, &output, false);

  auto out_flat = output.flat<TypeParam>();
  for (int i = 0; i < m * n; ++i) {
    EXPECT_NEAR(out_flat(i), kExpected, 1e-6)
        << "Output at index " << i << " is not close to expected precision.";
  }
}

REGISTER_TYPED_TEST_SUITE_P(MatMulOpTest,
                            MatMul_256x256x256,
                            MatMul_1x256x256,
                            MatMul_256x256x1,
                            MatMul_1x256x1,
                            // 中等规模
                            MatMul_5530x104x32,
                            MatMul_5530x116x32,
                            MatMul_5530x32x16,
                            MatMul_5530x16x1,
                            MatMul_7000x104x32,
                            MatMul_7000x116x32,
                            MatMul_7000x32x16,
                            MatMul_7000x16x1,
                            // 边界测试
                            MatMul_0x256x256,
                            MatMul_256x0x256,
                            MatMul_256x256x0,
                            MatMul_0x0x0,
                            MatMul_0x0x1,
                            MatMul_0x1x0,
                            MatMul_1x0x0,
                            MatMul_257x257x257,
                            MatMul_250x240x230,
                            MatMul_123x456x789,
                            MatMul_64x8192x64,
                            MatMul_256x1x256,
                            MatMul_64x4096x512,
                            MatMul_1x512x1000,
                            MatMul_1024x256x1,
                            MatMul_1x1x1,
                            MatMul_1x1x64,
                            MatMul_64x1x1,
                            MatMul_ZeroMatrix_A,
                            MatMul_ZeroMatrix_B,
                            MatMul_ZeroMatrix_AB,
                            MatMul_IdentityMatrix_A,
                            MatMul_IdentityMatrix_B,
                            MatMul_IdentityMatrix_AB,
                            MatMul_Inf_Positive,
                            MatMul_Inf_Negative,
                            MatMul_NaN,
                            MatMul_FloatMax_Times_One,
                            MatMul_FloatMin_Times_One,
                            MatMul_FloatMax_Minus_One_Times_One,
                            MatMul_FloatMin_Plus_One_Times_One,
                            MatMul_FloatMax_Half_Times_Two,
                            MatMul_FloatMin_Half_Times_Two,
                            MatMul_FloatMax_Minus_One_Half_Times_Two,
                            MatMul_FloatMin_Plus_One_Half_Times_Two,
                            MatMul_PrecisionSensitive);


REGISTER_TYPED_TEST_SUITE_P(FusedMatMulWithBiasOpTest,        //
                            MatMul256x256x256,                //
                            MatMul1x256x256,                  //
                            MatMul256x256x1,                  //
                            MatMul1x256x1,                    //
                            MatMul256x256x256WithActivation,  //
                            MatMul1x256x256WithActivation,    //
                            MatMul256x256x1WithActivation,    //
                            MatMul1x256x1WithActivation);

using MatMulTestTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, MatMulOpTest,
                               MatMulTestTypes);

// TODO(ezhulenev): Add support for more data types.
using FusedBiasAddDataTypes = ::testing::Types<float>;
INSTANTIATE_TYPED_TEST_SUITE_P(Test, FusedMatMulWithBiasOpTest,
                               FusedBiasAddDataTypes);

//----------------------------------------------------------------------------//
// Performance benchmarks are below.                                          //
//----------------------------------------------------------------------------//

template <typename T>
static Graph* Matmul(int m, int k, int n, bool transpose_a, bool transpose_b,
                     DataType type) {
  Graph* g = new Graph(OpRegistry::Global());
  Tensor in0(type, transpose_a ? TensorShape({k, m}) : TensorShape({m, k}));
  in0.flat<T>().setRandom();
  Tensor in1(type, transpose_b ? TensorShape({n, k}) : TensorShape({k, n}));
  in1.flat<T>().setRandom();
  test::graph::Matmul(g, test::graph::Constant(g, in0),
                      test::graph::Constant(g, in1), transpose_a, transpose_b);
  return g;
}

#define BM_MatmulDev(M, K, N, TA, TB, T, TFTYPE, DEVICE)                       \
  static void BM_Matmul##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE( \
      int iters) {                                                             \
    testing::UseRealTime();                                                    \
    testing::ItemsProcessed(static_cast<int64>(iters) * M * K * N * 2);        \
    test::Benchmark(#DEVICE, Matmul<T>(M, K, N, TA, TB, TFTYPE)).Run(iters);   \
  }                                                                            \
  BENCHMARK(BM_Matmul##_##M##_##K##_##N##_##TA##_##TB##_##TFTYPE##_##DEVICE);

#ifdef GOOGLE_CUDA

#define BM_Matmul(M, K, N, TA, TB)                                       \
  BM_MatmulDev(M, K, N, TA, TB, float, DT_FLOAT, cpu);                   \
  BM_MatmulDev(M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64, cpu); \
  BM_MatmulDev(M, K, N, TA, TB, float, DT_FLOAT, gpu);                   \
  BM_MatmulDev(M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64, gpu); \
  /* Uncomment to enable benchmarks for double/complex128: */            \
  // BM_MatmulDev(M, K, N, TA, TB, double, DT_DOUBLE, cpu);                   \
// BM_MatmulDev(M, K, N, TA, TB, std::complex<double>, DT_COMPLEX128, cpu); \
// BM_MatmulDev(M, K, N, TA, TB, double, DT_DOUBLE, gpu);                   \
// BM_MatmulDev(M, K, N, TA, TB, std::complex<double>, DT_COMPLEX128, gpu);

#else

#define BM_Matmul(M, K, N, TA, TB)                     \
  BM_MatmulDev(M, K, N, TA, TB, float, DT_FLOAT, cpu); \
  BM_MatmulDev(M, K, N, TA, TB, std::complex<float>, DT_COMPLEX64, cpu);

#endif  // GOOGLE_CUDA

// Batch size of 1 included for inference.
// Typical fully connected layers
BM_Matmul(1, 512, 512, false, false);
BM_Matmul(8, 512, 512, false, false);
BM_Matmul(16, 512, 512, false, false);
BM_Matmul(128, 512, 512, false, false);

BM_Matmul(1, 1024, 1024, false, false);
BM_Matmul(8, 1024, 1024, false, false);
BM_Matmul(16, 1024, 1024, false, false);
BM_Matmul(128, 1024, 1024, false, false);
BM_Matmul(4096, 4096, 4096, false, false);

// Backward for fully connected layers
BM_Matmul(1, 1024, 1024, false, true);
BM_Matmul(8, 1024, 1024, false, true);
BM_Matmul(16, 1024, 1024, false, true);
BM_Matmul(128, 1024, 1024, false, true);

// Forward softmax with large output size
BM_Matmul(1, 200, 10000, false, false);
BM_Matmul(8, 200, 10000, false, false);
BM_Matmul(20, 200, 10000, false, false);
BM_Matmul(20, 200, 20000, false, false);

// Backward softmax with large output size
BM_Matmul(1, 10000, 200, false, true);
BM_Matmul(1, 10000, 200, false, false);
BM_Matmul(8, 10000, 200, false, true);
BM_Matmul(20, 10000, 200, false, true);
BM_Matmul(20, 20000, 200, false, true);

// Test some matrix-vector multiplies.
BM_Matmul(50, 50, 1, false, false);
BM_Matmul(50, 50, 1, true, false);
BM_Matmul(50, 50, 1, false, true);
BM_Matmul(50, 50, 1, true, true);
BM_Matmul(500, 500, 1, false, false);
BM_Matmul(500, 500, 1, true, false);
BM_Matmul(500, 500, 1, false, true);
BM_Matmul(500, 500, 1, true, true);
BM_Matmul(2000, 2000, 1, false, false);
BM_Matmul(2000, 2000, 1, true, false);
BM_Matmul(2000, 2000, 1, false, true);
BM_Matmul(2000, 2000, 1, true, true);

// Test some vector-matrix multiplies.
BM_Matmul(1, 50, 50, false, false);
BM_Matmul(1, 50, 50, true, false);
BM_Matmul(1, 50, 50, false, true);
BM_Matmul(1, 50, 50, true, true);
BM_Matmul(1, 500, 500, false, false);
BM_Matmul(1, 500, 500, true, false);
BM_Matmul(1, 500, 500, false, true);
BM_Matmul(1, 500, 500, true, true);
BM_Matmul(1, 2000, 2000, false, false);
BM_Matmul(1, 2000, 2000, true, false);
BM_Matmul(1, 2000, 2000, false, true);
BM_Matmul(1, 2000, 2000, true, true);

// Test some rank-one products.
BM_Matmul(50, 1, 50, false, false);
BM_Matmul(50, 1, 50, true, false);
BM_Matmul(50, 1, 50, false, true);
BM_Matmul(50, 1, 50, true, true);
BM_Matmul(500, 1, 500, false, false);
BM_Matmul(500, 1, 500, true, false);
BM_Matmul(500, 1, 500, false, true);
BM_Matmul(500, 1, 500, true, true);
BM_Matmul(2000, 1, 2000, false, false);
BM_Matmul(2000, 1, 2000, true, false);
BM_Matmul(2000, 1, 2000, false, true);
BM_Matmul(2000, 1, 2000, true, true);

}  // end namespace tensorflow
