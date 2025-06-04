load("//tensorflow:tensorflow.default.bzl", "tf_kernel_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
)

DYNAMIC_DEPS = [
    "//tensorflow/core/framework:bounds_check",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
]

tf_kernel_library(
    name = "fused_embedding_ops",
    srcs = [
        "embedding_lookup_op.cc",
    ],
    deps = [
        "@eigen_archive//:eigen3",
        "@ktfop_archive//:ktfop",
    ] + DYNAMIC_DEPS
)

tf_kernel_library(
    name = "softmax_ops",
    srcs = [
        "softmax.cc",
    ],
    deps = [
        "@eigen_archive//:eigen3",
        "@ktfop_archive//:ktfop",
    ] + DYNAMIC_DEPS
)