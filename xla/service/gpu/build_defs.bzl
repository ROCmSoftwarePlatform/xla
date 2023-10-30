""" GPU-specific build macros.
"""

load("@local_config_cuda//cuda:build_defs.bzl", 
            "if_cuda_is_configured", "cuda_library")
load("@local_config_rocm//rocm:build_defs.bzl", 
            "if_rocm_is_configured", "rocm_copts")  # copybara:comment

def get_cub_sort_kernel_types(name = ""):
    """ List of supported types for CUB sort kernels.
    """
    return [
        "f16",
        "f32",
        "f64",
        "s8",
        "s16",
        "s32",
        "s64",
        "u8",
        "u16",
        "u32",
        "u64",
        "u16_b16",
        "u16_b32",
        "u16_b64",
        "u32_b16",
        "u32_b32",
        "u32_b64",
        "u64_b16",
        "u64_b32",
        "u64_b64",
    ]

def build_cub_sort_kernels(name, types, local_defines = [], **kwargs):
    """ Create build rules for all CUB sort kernels.
    """
    for suffix in types:
        gpu_kernel_library(
            name = name + "_" + suffix,
            local_defines = local_defines + ["CUB_TYPE_" + suffix.upper()],
            **kwargs
        )

def gpu_kernel_library(name, copts = [], local_defines = [], **kwargs):
    cuda_library(
        name = name,
        local_defines = local_defines + if_cuda_is_configured(["GOOGLE_CUDA=1"]) + 
                                        if_rocm_is_configured(["TENSORFLOW_USE_ROCM=1"]),
        copts = copts + rocm_copts(),  # copybara:comment
        **kwargs
    )