licenses(["notice"])

cc_library(
    name = "libkblas",
    srcs = ["lib/libkblas.so"],
    hdrs = [
        "include/kblas.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libktfop",
    srcs = ["lib/libktfop.so"],
    hdrs = [
        "include/ktfop.h",
        "math/BinaryOp.h",
        "math/tools.h",
    ],
    includes = ["."],
    deps = [":libkblas"],
    visibility = ["//visibility:public"],
)
