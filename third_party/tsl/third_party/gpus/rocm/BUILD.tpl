load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@local_config_rocm//rocm:build_defs.bzl", "rocm_version_number", "select_threshold")

licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "using_hipcc",
    values = {
        "define": "using_rocm_hipcc=true",
    },
)

cc_library(
    name = "rocm_config",
    hdrs = [
        "rocm_config/rocm_config.h",
    ],
    include_prefix = "rocm",
    strip_include_prefix = "rocm_config",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rocm_headers",
    hdrs = glob([
        "%{rocm_root}/include/**",
        "%{rocm_root}/lib/llvm/lib/**/*.h",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
        "%{rocm_root}/include/rocrand",
        "%{rocm_root}/include/roctracer",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hip",
    hdrs = glob(["%{rocm_root}/include/hip/**"]),
    data = glob(["%{rocm_root}/lib/hip/**"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkstatic = 1,
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "rocblas",
    srcs = glob(["%{rocm_root}/lib/librocblas*.so*"]),
    hdrs = glob(["%{rocm_root}/include/rocblas/**"]),
    data = glob([
        "%{rocm_root}/lib/librocblas*.so",
        "%{rocm_root}/lib/rocblas/**",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkstatic = 1,
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "rocfft",
    srcs = glob(["%{rocm_root}/lib/librocfft*.so*"]),
    data = glob(["%{rocm_root}/lib/librocfft*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkstatic = 1,
    deps = [":rocm_config"],
)

cc_library(
    name = "hipfft",
    srcs = glob(["%{rocm_root}/lib/libhipfft*.so*"]),
    data = glob(["%{rocm_root}/lib/libhipfft*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkstatic = 1,
    deps = [":rocm_config"],
)

cc_library(
    name = "hiprand",
    srcs = glob(["%{rocm_root}/lib/libhiprand*.so*"]),
    hdrs = glob(["%{rocm_root}/include/hiprand/**"]),
    data = glob(["%{rocm_root}/lib/libhiprand*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
        "%{rocm_root}/include/rocrand",
    ],
    linkstatic = 1,
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "miopen",
    srcs = glob(["%{rocm_root}/lib/libMIOpen*.so*"]),
    hdrs = glob(["%{rocm_root}/include/rccl/**"]),
    data = glob(["%{rocm_root}/lib/libMIOpen*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkstatic = 1,
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "rccl",
    srcs = glob(["%{rocm_root}/lib/librccl*.so*"]),
    hdrs = glob(["%{rocm_root}/include/rccl/**"]),
    data = glob(["%{rocm_root}/lib/librccl*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include",
    ],
    linkstatic = 1,
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "rocm",
    visibility = ["//visibility:public"],
    deps = [
        ":hip",
        ":hipblas",
        ":hiprand",
        ":hipsolver",
        ":hipsparse",
        ":llvm",
        ":miopen",
        ":rocblas",
        ":rocm_config",
        ":rocsolver",
        ":roctracer",
    ] + select_threshold(
        above_or_eq = [":hipfft"],
        below = [":rocfft"],
        threshold = 40100,
        value = rocm_version_number(),
    ),
)

filegroup(
    name = "rocm_bin",
    srcs = glob(["%{rocm_root}/bin/**/*"]),
    visibility = ["//visibility:public"],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
)

cc_library(
    name = "rocprim",
    srcs = [
        "%{rocm_root}/include/hipcub/hipcub_version.hpp",
        "%{rocm_root}/include/rocprim/rocprim_version.hpp",
    ],
    hdrs = glob([
        "%{rocm_root}/include/hipcub/**",
        "%{rocm_root}/include/rocprim/**",
    ]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/hipcub",
        "%{rocm_root}/include/rocprim",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm",
    ],
)

cc_library(
    name = "hipsparse",
    srcs = glob(["%{rocm_root}/lib/libhipsparse*.so*"]),
    hdrs = glob(["%{rocm_root}/include/hipsparse/**"]),
    data = glob(["%{rocm_root}/lib/libhipsparse*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "roctracer",
    hdrs = glob(["%{rocm_root}/include/roctracer/**"]),
    data = glob(["%{rocm_root}/lib/libroctracer*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "rocsolver",
    srcs = glob(["%{rocm_root}/lib/librocsolver*.so*"]),
    hdrs = glob(["%{rocm_root}/include/rocsolver/**"]),
    data = glob(["%{rocm_root}/lib/librocsolver*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "hipsolver",
    srcs = glob(["%{rocm_root}/lib/libhipsolver*.so*"]),
    hdrs = glob(["%{rocm_root}/include/hipsolver/**"]),
    data = glob(["%{rocm_root}/lib/libhipsolver*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "hipblas",
    srcs = glob(["%{rocm_root}/lib/libhipblas*.so*"]),
    hdrs = glob(["%{rocm_root}/include/hipblas/**"]),
    data = glob(["%{rocm_root}/lib/libhipblas*.so*"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "rocrand",
    srcs = glob(["%{rocm_root}/lib/librocrand*.so*"]),
    hdrs = glob(["%{rocm_root}/include/rocrand/**"]),
    include_prefix = "rocm",
    includes = [
        "%{rocm_root}/include/",
    ],
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [":rocm_config"],
)

cc_library(
    name = "llvm",
    srcs = select({
        "@platforms//cpu:x86_64": glob(["%{rocm_root}/lib/llvm/lib/**/*x86_64.so*"]),
        "@platforms//cpu:x86_32": glob(["%{rocm_root}/lib/llvm/lib/**/*i386.so*"]),
        "//conditions:default": glob(["%{rocm_root}/lib/llvm/lib/**/*x86_64.so*"]),
    }),
    hdrs = glob(["%{rocm_root}/lib/llvm/lib/**/*.h"]),
    data = glob(["%{rocm_root}/lib/llvm/**"]),
    include_prefix = "rocm",
    strip_include_prefix = "%{rocm_root}",
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_config",
    ],
)

filegroup(
    name = "rocm_root",
    srcs = [
        "%{rocm_root}/bin/clang-offload-bundler",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["%{rocm_root}/**"]),
)
