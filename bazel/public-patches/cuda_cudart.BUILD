load("@rules_cc//cc:cc_library.bzl", "cc_library")

# CUDA Driver API header (cuda.h) from the NVIDIA cuda_cudart redistributable.
cc_library(
    name = "cuda_headers",
    hdrs = ["include/cuda.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

# The cuda_cudart redistributable does not ship crt/host_defines.h
# (a compiler-internal header only present in the full CUDA toolkit).
# device_types.h and other runtime headers use #include "crt/host_defines.h",
# so we generate a minimal stub for host-only (non-nvcc) compilation.
# This mirrors what the full CUDA SDK's crt/host_defines.h defines for
# __GNUC__ (clang/GCC) and !__CUDACC__ (host-only) paths.
genrule(
    name = "crt_host_defines_h",
    outs = ["include/crt/host_defines.h"],
    cmd = r"""cat > $@ << 'EOF'
#pragma once
#ifndef __CUDA_CRT_HOST_DEFINES_H__
#define __CUDA_CRT_HOST_DEFINES_H__

/* Attribute macros for GCC/Clang (host compiler path) */
#if defined(__GNUC__)
#define __align__(n) __attribute__((aligned(n)))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __no_return__ __attribute__((noreturn))
#define __annotate__(a) __attribute__((a))
#define __location__(a) __annotate__(a)
#define CUDARTAPI
#define __builtin_align__(a) __align__(a)
#endif /* __GNUC__ */

/* CUDA qualifier macros -- empty for host-only (non-nvcc) compilation */
#if !defined(__CUDACC__)
#define __device_builtin__
#define __device_builtin_texture_type__
#define __device_builtin_surface_type__
#define __cudart_builtin__
#define __host__
#define __device__
#define __global__
#define __shared__
#define __constant__
#define __managed__
#define __launch_bounds__(...)
#endif /* !__CUDACC__ */

#endif /* __CUDA_CRT_HOST_DEFINES_H__ */
EOF
""",
)

# The cuda_cudart redistributable ships include/host_config.h as a forwarder
# to crt/host_config.h, but the latter is only present in the full CUDA
# toolkit. The actual content of crt/host_config.h is compiler/glibc
# version checks and compiler-internal macros used by nvcc; for host-only
# (non-nvcc) compilation an empty stub is sufficient to let cuda_runtime.h
# include it without errors.
genrule(
    name = "crt_host_config_h",
    outs = ["include/crt/host_config.h"],
    cmd = r"""cat > $@ << 'EOF'
#pragma once
#ifndef __CUDA_CRT_HOST_CONFIG_H__
#define __CUDA_CRT_HOST_CONFIG_H__
/* Empty stub for host-only (non-nvcc) compilation. */
#endif /* __CUDA_CRT_HOST_CONFIG_H__ */
EOF
""",
)

# CUDA runtime headers (no libcudart.so) from the redistributable.
#
# This target exposes cuda_runtime_api.h, cuda_runtime.h, and all other
# headers WITHOUT linking against libcudart.so.  Consumers that need
# cudaDeviceSynchronize (e.g. profiler backends) should depend on the
# on-demand dlopen stub (libcudart_stub) instead of linking libcudart.so
# directly.
cc_library(
    name = "cuda_runtime_headers",
    hdrs = glob(
        [
            "include/**/*.h",
            "include/**/*.hpp",
        ],
        # cuda.h is exposed by :cuda_headers; avoid duplicate header warnings.
        exclude = ["include/cuda.h"],
    ) + [
        ":crt_host_config_h",
        ":crt_host_defines_h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [":cuda_headers"],
)
