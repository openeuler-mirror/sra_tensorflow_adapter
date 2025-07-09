licenses(["notice"])

cc_library(
    name = "kdnn",
    hdrs = glob(["kdnn_include/**/*.hpp"]),
    includes = ["kdnn_include"],
    srcs = ["kdnn_lib/libkdnn.so"],
    copts = ["-fopenmp"],
    linkopts = ["-fopenmp"],
    alwayslink = 1,
)

cc_library(
    name = "kdnn_adapter",
    hdrs = ["kdnn_adapter.h", "kdnn_threadpool.h"],
    strip_include_prefix = "/third_party/KDNN",
    visibility = ["//visibility:public"],
    deps = [":kdnn"],
)
