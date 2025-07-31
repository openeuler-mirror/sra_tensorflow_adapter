licenses(["notice"])

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

config_setting(
    name = "enable_kdnn",
    define_values = {
        "enable_kdnn": "true",
    },
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kdnn",
    hdrs = glob(["src/dnn/include/**/*.hpp"]),
    includes = ["src/dnn/include"],
    srcs = glob(["src/dnn/src/*.cpp", "src/dnn/src/*.hpp"]),
)

cc_library(
    name = "ksparse",
    hdrs = glob([
        "src/dnn/include/**/*.h",
        "src/dnn/src/sparse/**/*.S",
        "src/dnn/src/sparse/**/*.h",
        "src/dnn/src/sparse/**/*.c",
        ]),
    includes = [
        "src/dnn/include",
        "src/dnn/include/internal",
        "src/dnn/include/internal/common",
        "src/dnn/include/internal/driver",
        "src/dnn/include/internal/format",
        "src/dnn/include/internal/kernel",
        "src/dnn/include/internal/kernel/level3",
        "src/dnn/src/sparse/kernel/",
        "src/dnn/src/sparse/interface/",
        ],
    srcs = glob([
        "src/dnn/src/sparse/common/*.c", 
        "src/dnn/src/sparse/driver/*.c", 
        "src/dnn/src/sparse/interface/*.c", 
        "src/dnn/src/sparse/kernel/gemm_csr_k1x1.c", 
        "src/dnn/src/sparse/kernel/gemm_csr_k1x1_has_beta.c", 
        "src/dnn/src/sparse/kernel/gemm_csr_kernel.c", 
        "src/dnn/src/sparse/kernel/vector_scal_kernel.c", 
        "src/dnn/src/sparse/kernel/gemm_csr_row_k_small.c", 
        "src/dnn/src/sparse/**/*.S",
        ]),
)

cc_library(
    name = "kdnn_adapter",
    hdrs = ["kdnn_adapter.h", "kdnn_threadpool.h"],
    strip_include_prefix = "/third_party/KDNN",
    visibility = ["//visibility:public"],
    deps = [":kdnn", ":ksparse"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)