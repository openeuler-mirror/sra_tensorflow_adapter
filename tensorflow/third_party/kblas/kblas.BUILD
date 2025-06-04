cc_import(
    name = "kblas_so",
    shared_library = "lib/sve/kblas/locking/libkblas.so",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "kblas",
    hdrs = ["include/kblas.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)