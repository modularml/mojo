# Defines the imported pre-built UCX libraries.

load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Import rules for each shared library found in the pre-built archive.
# Bazel uses the static_library for linking when interface_library is absent.
# NOTE: most UCX libs have static constructors and destructors and hence
# require `alwayslink = True`.
# This ensures that those static ctors and dtors are linked in and run.
cc_import(
    name = "ucp_import",
    static_library = "lib/libucp.a",
    alwayslink = True,
)

cc_import(
    name = "ucs_import",
    static_library = "lib/libucs.a",
    alwayslink = True,
)

cc_import(
    name = "ucs_signal_import",
    static_library = "lib/libucs_signal.a",
    alwayslink = True,
)

cc_import(
    name = "uct_import",
    static_library = "lib/libuct.a",
    alwayslink = True,
)

cc_import(
    name = "ucm_import",
    static_library = "lib/libucm.a",
    alwayslink = True,
)

cc_import(
    name = "uct_cma_import",
    static_library = "lib/ucx/libuct_cma.a",
    alwayslink = True,
)

# cc_import(
#     name = "uct_cuda_gdrcopy_import",
#     static_library = "lib/ucx/libuct_cuda_gdrcopy.a",
#     alwayslink = True,
# )

cc_import(
    name = "uct_cuda_import",
    static_library = "lib/ucx/libuct_cuda.a",
    alwayslink = True,
)

cc_import(
    name = "ucm_cuda_import",
    static_library = "lib/ucx/libucm_cuda.a",
    alwayslink = True,
)

cc_import(
    name = "uct_rocm_import",
    static_library = "lib/ucx/libuct_rocm.a",
    alwayslink = True,
)

cc_import(
    name = "ucm_rocm_import",
    static_library = "lib/ucx/libucm_rocm.a",
    alwayslink = True,
)

cc_import(
    name = "uct_ib_import",
    static_library = "lib/ucx/libuct_ib.a",
    alwayslink = True,
)

cc_import(
    name = "uct_ib_mlx5_import",
    static_library = "lib/ucx/libuct_ib_mlx5.a",
    alwayslink = True,
)

# Top-level libraries target that bundle the headers and imported libraries.
# Targets depending on UCX should depend on one of these targets.
cc_library(
    name = "ucx_cpu",
    hdrs = glob(["include/**/*"]),
    # Specifies the include path relative to the repository root.
    includes = ["include"],
    deps = [
        ":ucm_import",
        ":ucp_import",
        ":ucs_import",
        ":ucs_signal_import",
        ":uct_cma_import",
        ":uct_import",
    ],
)

cc_library(
    name = "ucx_cuda",
    hdrs = glob(["include/**/*"]),
    # Specifies the include path relative to the repository root.
    includes = ["include"],
    deps = [
        ":ucm_import",
        ":ucm_cuda_import",
        ":ucp_import",
        ":ucs_import",
        ":ucs_signal_import",
        ":uct_cma_import",
        # ":uct_cuda_gdrcopy_import",
        ":uct_cuda_import",
        ":uct_import",
    ],
)

cc_library(
    name = "ucx_rocm",
    hdrs = glob(["include/**/*"]),
    # Specifies the include path relative to the repository root.
    includes = ["include"],
    deps = [
        ":ucm_import",
        ":ucm_rocm_import",
        ":ucp_import",
        ":ucs_import",
        ":ucs_signal_import",
        ":uct_cma_import",
        ":uct_import",
        ":uct_rocm_import",
    ],
)

# CPU + verbs flavor: the cpu flavor plus the uct_ib RDMA transports, for a
# CPU-only process that still transfers over an InfiniBand fabric — dKV, which
# runs on GPU hosts but registers host DRAM only. The plain cpu flavor has
# tcp/shm/cma alone and would fall back to TCP on such a host.
cc_library(
    name = "ucx_cpu_verbs",
    hdrs = glob(["include/**/*"]),
    # Specifies the include path relative to the repository root.
    includes = ["include"],
    deps = [
        ":ucm_import",
        ":ucp_import",
        ":ucs_import",
        ":ucs_signal_import",
        ":uct_cma_import",
        ":uct_ib_import",
        ":uct_ib_mlx5_import",
        ":uct_import",
    ],
)

cc_library(
    name = "ucx_cuda_verbs",
    hdrs = glob(["include/**/*"]),
    # Specifies the include path relative to the repository root.
    includes = ["include"],
    deps = [
        ":ucm_import",
        ":ucm_cuda_import",
        ":ucp_import",
        ":ucs_import",
        ":ucs_signal_import",
        ":uct_cma_import",
        # ":uct_cuda_gdrcopy_import",
        ":uct_cuda_import",
        ":uct_import",
        ":uct_ib_import",
        ":uct_ib_mlx5_import",
    ],
)

cc_library(
    name = "ucx_rocm_verbs",
    hdrs = glob(["include/**/*"]),
    # Specifies the include path relative to the repository root.
    includes = ["include"],
    deps = [
        ":ucm_import",
        ":ucm_rocm_import",
        ":ucp_import",
        ":ucs_import",
        ":ucs_signal_import",
        ":uct_cma_import",
        ":uct_ib_import",
        ":uct_ib_mlx5_import",
        ":uct_import",
        ":uct_rocm_import",
    ],
)
