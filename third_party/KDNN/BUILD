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
    name = "kdnn_adapter",
    hdrs = ["kdnn_adapter.h", "kdnn_threadpool.h"],
    strip_include_prefix = "/third_party/KDNN",
    visibility = ["//visibility:public"],
    deps = [":kdnn"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
)