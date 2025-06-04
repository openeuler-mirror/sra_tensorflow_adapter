cc_import(
    name = "ktfop_so",
    shared_library = "lib/sve/libktfop.so",
    deps = ["@kblas_archive//:kblas_so"],
)

cc_library(
    name = "ktfop",
    hdrs = ["include/ktfop.h"],
    includes = ["include"],
    deps = [":ktfop_so",
            "@kblas_archive//:kblas"],
    visibility = ["//visibility:public"],
)
