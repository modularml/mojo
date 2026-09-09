# Bazel overlay for vendored upstream NIXL (github.com/ai-dynamo/nixl).
#
# Upstream uses Meson; this BUILD compiles the unmodified upstream sources
# into the same shape the MLRT consumers used to provide. Public headers are
# exposed via the cc_library `nixl` and consumed with `#include "nixl.h"`,
# the upstream convention. UCX/libfabric backends are built as shared plugins
# named `libplugin_<NAME>.so` to match upstream's `dlopen()` plugin discovery.
#
# Etcd-backed listener (HAVE_ETCD) is left disabled — we don't link
# `etcd-cpp-api` here.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_pkg//pkg:mappings.bzl", "pkg_files", "strip_prefix")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

package(default_visibility = ["//visibility:public"])

# Upstream NIXL is Linux-only (uses SOCK_NONBLOCK, dlopen, etc.). Apply the
# constraint at the package level so every target inherits it.
_LINUX_X86 = [
    "@platforms//cpu:x86_64",
    "@platforms//os:linux",
]

# --- Public API headers ---------------------------------------------------
# Consumers use `#include "nixl.h"` and friends. The `includes` attribute on
# the downstream cc_library makes these visible without rewriting upstream
# code.
cc_library(
    name = "nixl_api_headers",
    hdrs = glob([
        "src/api/cpp/*.h",
        "src/api/cpp/backend/*.h",
        "src/api/cpp/telemetry/*.h",
    ]) + [
        # backend_engine.h reaches into core/telemetry for telemetry_event.h.
        "src/core/telemetry/telemetry_event.h",
    ],
    includes = [
        "src/api/cpp",
        "src/api/cpp/backend",
        "src/api/cpp/telemetry",
        "src/core/telemetry",
    ],
    # Public headers compile on any platform; the Linux constraint is on the
    # library/plugin targets that pull in SOCK_NONBLOCK, dlopen, etc.
)

# --- Internal common utilities (logging, config, hw_info, uuid) -----------
# Upstream meson builds these into libnixl_common as a shared library that
# links absl::log/strings/etc. We keep it as a regular cc_library; the
# resulting object code lands in the consumer shared library.
cc_library(
    name = "nixl_common",
    srcs = [
        "src/utils/common/configuration.cpp",
        "src/utils/common/hw_info.cpp",
        "src/utils/common/nixl_log.cpp",
        "src/utils/common/uuid_v4.cpp",
    ],
    hdrs = glob([
        "src/utils/common/*.h",
        "src/utils/common/*.tpp",
    ]),
    copts = [
        # Upstream's meson defines these on the command line.
        '-DNIXL_VERSION=\\"1.3.0\\"',
        '-DNIXL_GIT_HASH=\\"upstream-v1.3.0\\"',
    ],
    # Upstream's meson sets utils_inc_dirs=src/utils so `#include "common/..."`
    # works; but telemetry.cpp also uses bare `#include "util.h"` which
    # requires `src/utils/common` to be on the path too.
    includes = [
        "src/utils",
        "src/utils/common",
    ],
    target_compatible_with = _LINUX_X86,
    deps = [
        ":nixl_api_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/log:globals",
        "@abseil-cpp//absl/log:initialize",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@tomlplusplus//:toml++-src",
    ],
    alwayslink = True,
)

cc_library(
    name = "nixl_serdes",
    srcs = ["src/utils/serdes/serdes.cpp"],
    hdrs = ["src/utils/serdes/serdes.h"],
    includes = ["src/utils"],
    target_compatible_with = _LINUX_X86,
    deps = [
        ":nixl_api_headers",
        ":nixl_common",
    ],
    # alwayslink so libplugin_UCX.so re-exports the full symbol set even when
    # its own code doesn't reference every helper. Without this, --gc-sections
    # strips template instantiations and downstream nanobind bindings hit
    # undefined-symbol ImportErrors at runtime.
    alwayslink = True,
)

cc_library(
    name = "nixl_stream",
    srcs = ["src/utils/stream/metadata_stream.cpp"],
    hdrs = ["src/utils/stream/metadata_stream.h"],
    includes = ["src/utils"],
    target_compatible_with = _LINUX_X86,
    deps = [
        ":nixl_api_headers",
        ":nixl_common",
    ],
    alwayslink = True,
)

# --- Infrastructure (descriptors and memory section) ----------------------
# Upstream meson: nixl_build_lib in src/infra/meson.build.
cc_library(
    name = "nixl_infra",
    srcs = [
        "src/infra/nixl_descriptors.cpp",
        "src/infra/nixl_memory_section.cpp",
    ],
    hdrs = [
        "src/infra/mem_section.h",
        "src/infra/test_utils.h",
    ],
    includes = ["src/infra"],
    target_compatible_with = _LINUX_X86,
    deps = [
        ":nixl_api_headers",
        ":nixl_common",
        ":nixl_serdes",
        "@abseil-cpp//absl/strings",
    ],
    # nixl_descriptors.cpp explicitly instantiates nixlDescList<...> for the
    # public Desc types. alwayslink keeps those instantiations in
    # libplugin_UCX.so even when its own code doesn't call every method, so
    # Python bindings can resolve them at dlopen time.
    alwayslink = True,
)

# --- Core (agent, plugin manager, listener, telemetry runtime) ------------
# Upstream meson: nixl_lib in src/core/meson.build. Compiled together with
# the telemetry .cpp files that live under src/core/telemetry/.
cc_library(
    name = "nixl",
    srcs = [
        "src/core/nixl_agent.cpp",
        "src/core/nixl_enum_strings.cpp",
        "src/core/nixl_listener.cpp",
        "src/core/nixl_plugin_manager.cpp",
        "src/core/telemetry/buffer_exporter.cpp",
        "src/core/telemetry/buffer_plugin.cpp",
        # nop_plugin.cpp defines createStaticNOPPlugin(), which
        # nixl_plugin_manager.cpp references unconditionally via
        # registerBuiltinPlugins() since upstream v1.3.0. Without it the
        # consumer link fails with an undefined reference.
        "src/core/telemetry/nop_plugin.cpp",
        "src/core/telemetry/telemetry.cpp",
    ],
    hdrs = glob([
        "src/core/*.h",
        "src/core/telemetry/*.h",
    ]),
    includes = [
        "src/core",
        "src/core/telemetry",
    ],
    # Upstream meson links -lstdc++fs for <filesystem>; modern libstdc++ has
    # this in the main library, so we omit it.
    target_compatible_with = _LINUX_X86,
    deps = [
        ":nixl_api_headers",
        ":nixl_common",
        ":nixl_infra",
        ":nixl_serdes",
        ":nixl_stream",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@asio",
    ],
    alwayslink = True,
)

# --- UCX backend plugin ---------------------------------------------------
# Upstream builds these as `libplugin_UCX.so` etc. We split the cc_library
# (compilable object) from the cc_binary (resulting `.so`) so multiple
# UCX flavors (cpu/cuda/cuda_verbs/rocm/rocm_verbs) can reuse the sources.

filegroup(
    name = "ucx_plugin_srcs",
    srcs = [
        "src/plugins/ucx/config.cpp",
        "src/plugins/ucx/config.h",
        "src/plugins/ucx/mem_list.cpp",
        "src/plugins/ucx/mem_list.h",
        "src/plugins/ucx/rkey.cpp",
        "src/plugins/ucx/rkey.h",
        "src/plugins/ucx/ucx_backend.cpp",
        "src/plugins/ucx/ucx_backend.h",
        "src/plugins/ucx/ucx_enums.cpp",
        "src/plugins/ucx/ucx_enums.h",
        "src/plugins/ucx/ucx_plugin.cpp",
        "src/plugins/ucx/ucx_utils.cpp",
        "src/plugins/ucx/ucx_utils.h",
    ],
)

cc_library(
    name = "ucx_plugin_lib_cpu",
    srcs = [":ucx_plugin_srcs"],
    target_compatible_with = _LINUX_X86,
    deps = [
        ":nixl",
        ":nixl_api_headers",
        ":nixl_common",
        ":nixl_serdes",
        "@abseil-cpp//absl/strings",
        "@asio",
        "@ucx_prebuilt//:ucx_cpu",
    ],
    alwayslink = True,
)

cc_library(
    name = "ucx_plugin_lib_cpu_verbs",
    srcs = [":ucx_plugin_srcs"],
    target_compatible_with = _LINUX_X86,
    deps = [
        ":nixl",
        ":nixl_api_headers",
        ":nixl_common",
        ":nixl_serdes",
        "@abseil-cpp//absl/strings",
        "@asio",
        "@ucx_prebuilt//:ucx_cpu_verbs",
    ],
    alwayslink = True,
)

cc_library(
    name = "ucx_plugin_lib_cuda",
    srcs = [":ucx_plugin_srcs"],
    target_compatible_with = _LINUX_X86,
    deps = [
        ":nixl",
        ":nixl_api_headers",
        ":nixl_common",
        ":nixl_serdes",
        "@abseil-cpp//absl/strings",
        "@asio",
        "@ucx_prebuilt//:ucx_cuda",
    ],
    alwayslink = True,
)

cc_library(
    name = "ucx_plugin_lib_cuda_verbs",
    srcs = [":ucx_plugin_srcs"],
    target_compatible_with = _LINUX_X86,
    deps = [
        ":nixl",
        ":nixl_api_headers",
        ":nixl_common",
        ":nixl_serdes",
        "@abseil-cpp//absl/strings",
        "@asio",
        "@ucx_prebuilt//:ucx_cuda_verbs",
    ],
    alwayslink = True,
)

cc_library(
    name = "ucx_plugin_lib_rocm",
    srcs = [":ucx_plugin_srcs"],
    target_compatible_with = _LINUX_X86,
    deps = [
        ":nixl",
        ":nixl_api_headers",
        ":nixl_common",
        ":nixl_serdes",
        "@abseil-cpp//absl/strings",
        "@asio",
        "@ucx_prebuilt//:ucx_rocm",
    ],
    alwayslink = True,
)

cc_library(
    name = "ucx_plugin_lib_rocm_verbs",
    srcs = [":ucx_plugin_srcs"],
    target_compatible_with = _LINUX_X86,
    deps = [
        ":nixl",
        ":nixl_api_headers",
        ":nixl_common",
        ":nixl_serdes",
        "@abseil-cpp//absl/strings",
        "@asio",
        "@ucx_prebuilt//:ucx_rocm_verbs",
    ],
    alwayslink = True,
)

# --- UCCL backend plugin --------------------------------------------------
# UCCL runs its transport protocol on CPUs over plain verbs queue pairs, so
# it saturates AMD RoCE NICs. The plugin source is vendored here; the transport
# itself is the prebuilt @uccl_prebuilt//:uccl (libuccl_p2p.so). Single AMD
# flavor: UCCL does its own transport, so there is no separate verbs variant.
cc_library(
    name = "uccl_plugin_lib",
    srcs = [
        "src/plugins/uccl/uccl_backend.cpp",
        "src/plugins/uccl/uccl_backend.h",
        "src/plugins/uccl/uccl_plugin.cpp",
    ],
    # Upstream uccl_backend.h does not mark its overrides consistently; relax
    # the Modular -Werror on this vendored third-party source.
    copts = ["-Wno-inconsistent-missing-override"],
    target_compatible_with = _LINUX_X86,
    deps = [
        ":nixl",
        ":nixl_api_headers",
        ":nixl_common",
        ":nixl_serdes",
        "@uccl_prebuilt//:uccl",
    ],
    alwayslink = True,
)

# libuccl_p2p.so is staged alongside libplugin_UCCL.so in the plugin dir, so
# the first rpath ($ORIGIN) resolves it; the second reaches <root>/lib for
# libnixl.so. -z undefs allows the transport lib's undefined Python symbols
# to bind at load time in the Python serving process.
cc_binary(
    name = "rocm-uccl/libplugin_UCCL.so",
    linkopts = [
        "-Wl,-z,undefs",
        "-Wl,-rpath,$$ORIGIN",
        # Installed layout: <root>/lib/nixl/rocm-uccl/ → <root>/lib.
        "-Wl,-rpath,$$ORIGIN/../../../lib",
    ],
    linkshared = True,
    linkstatic = True,
    target_compatible_with = _LINUX_X86,
    deps = [":uccl_plugin_lib"],
)

# Co-locate the transport lib next to the plugin in a `rocm-uccl/` runfiles
# subdir so the plugin's $ORIGIN rpath resolves it under `bazel run` (the NIXL
# drag race). Packaging stages the same pair into lib/nixl/rocm-uccl/ via its
# own path mapping (utils/packaging), which is why this copy is needed only for
# the runfiles layout.
genrule(
    name = "rocm-uccl-transport-copy",
    srcs = ["@uccl_prebuilt//:libuccl_p2p_so"],
    outs = ["rocm-uccl/libuccl_p2p.so"],
    cmd = "cp $< $@",
    target_compatible_with = _LINUX_X86,
)

# The loadable UCCL flavor for a runfiles tree: the plugin plus its co-located
# transport lib. Depend on this from a `bazel run` target that selects the
# UCCL backend on an AMD host (the drag race). Deliberately NOT part of the
# shared :nixl_host_plugins set — staging it there would make the default-UCCL
# resolver pick this flavor for the gating same-node AMD transfer tests, which
# do not set UCCL_P2P_DISABLE_IPC and would break.
filegroup(
    name = "rocm_uccl_host_plugin",
    srcs = [
        "rocm-uccl/libplugin_UCCL.so",
        "rocm-uccl/libuccl_p2p.so",
    ],
    target_compatible_with = _LINUX_X86,
)

# --- libfabric backend plugin --------------------------------------------
# Upstream src/utils/libfabric/* is a separate library that the plugin
# links against. We fold both into a single cc_library that builds the
# shared plugin.

filegroup(
    name = "libfabric_plugin_srcs",
    srcs = [
        "src/plugins/libfabric/libfabric_backend.cpp",
        "src/plugins/libfabric/libfabric_backend.h",
        "src/plugins/libfabric/libfabric_plugin.cpp",
        "src/utils/libfabric/libfabric_common.cpp",
        "src/utils/libfabric/libfabric_common.h",
        "src/utils/libfabric/libfabric_rail.cpp",
        "src/utils/libfabric/libfabric_rail.h",
        "src/utils/libfabric/libfabric_rail_manager.cpp",
        "src/utils/libfabric/libfabric_rail_manager.h",
        "src/utils/libfabric/libfabric_topology.cpp",
        "src/utils/libfabric/libfabric_topology.h",
    ],
)

cc_library(
    name = "libfabric_plugin_lib",
    srcs = [":libfabric_plugin_srcs"],
    copts = [
        "-DHAVE_LIBFABRIC",
        "-DHAVE_CUDA",
    ],
    target_compatible_with = _LINUX_X86,
    deps = [
        ":nixl",
        ":nixl_api_headers",
        ":nixl_common",
        ":nixl_serdes",
        "@abseil-cpp//absl/strings",
        "@cuda_x86_64//:cuda_headers",
        "@cuda_x86_64//:cuda_runtime_headers",
        "@efa_libfabric_prebuilt_cuda//:hwloc",
        "@efa_libfabric_prebuilt_cuda//:libfabric",
        "@efa_libfabric_prebuilt_cuda//:numa",
    ],
    alwayslink = True,
)

# CPU-only flavor of the libfabric plugin: no -DHAVE_CUDA, linked against the
# CUDA-free EFA libfabric prebuilt. The cuda prebuilt's libfabric.so.1 carries a
# hard DT_NEEDED on libcudart/libcuda/libnvidia-ml, so a CPU-only consumer (dKV,
# which registers host DRAM only and never requests FI_HMEM) must use this
# variant to load on a host without the CUDA driver stack. Mirrors the
# cpu/cuda split already used for the UCX plugin.
cc_library(
    name = "libfabric_plugin_lib_cpu",
    srcs = [":libfabric_plugin_srcs"],
    copts = [
        "-DHAVE_LIBFABRIC",
    ],
    target_compatible_with = _LINUX_X86,
    deps = [
        ":nixl",
        ":nixl_api_headers",
        ":nixl_common",
        ":nixl_serdes",
        "@abseil-cpp//absl/strings",
        "@efa_libfabric_prebuilt//:hwloc",
        "@efa_libfabric_prebuilt//:libfabric",
        "@efa_libfabric_prebuilt//:numa",
    ],
    alwayslink = True,
)

# --- Plugin .so outputs ---------------------------------------------------
# Upstream's plugin loader (nixl_plugin_manager.cpp) looks for files named
# `libplugin_<NAME>.so` where <NAME> matches the plugin id (UCX, LIBFABRIC).
# Use cc_binary with linkshared so Bazel emits exactly those filenames.
#
# The GPU-flavored plugins live in per-vendor cuda/ and rocm/ subdirectories
# (the slash in the target name is what creates the subdir): NIXL discovers
# plugins by the fixed filename in a single NIXL_PLUGIN_DIR, and one universal
# linux_x86_64 package serves both GPU vendors, so max._core points that var
# at the subdir matching the host GPU vendor. The plugin sources are
# GPU-vendor-agnostic; the flavor difference is which static UCX/libfabric
# gets folded in.
cc_binary(
    name = "cuda/libplugin_UCX.so",
    linkopts = [
        "-Wl,-z,undefs",
        # Installed layout: <root>/lib/nixl/cuda/ → <root>/lib.
        "-Wl,-rpath,$$ORIGIN/../../../lib",
    ],
    linkshared = True,
    linkstatic = True,
    target_compatible_with = _LINUX_X86,
    deps = [":ucx_plugin_lib_cuda"],
)

# CUDA + verbs flavor: a strict superset of the cuda flavor that adds the
# uct_ib RDMA transports for internode transfers over an InfiniBand fabric
# (UCX picks transports per connection at runtime — same-node peers still use
# cuda_ipc/shm). The plain cuda flavor above lacks uct_ib and falls back to
# TCP/IPoIB on IB hosts; the CUDA inter-node path historically went through
# libfabric/EFA on AWS, so verbs was never needed there. On a pure-IB fabric
# (no EFA) this flavor is what makes NIXL transfer RDMA. Like the rocm-verbs
# flavor it links libibverbs.so.1 alone, so its mlx5dv_* symbols rest on the
# RTLD_GLOBAL preload in _nixl_plugin_deps.py rather than on a DT_NEEDED;
# max._core selects this flavor only when both libibverbs.so.1 and
# libmlx5.so.1 resolve, and otherwise falls back to the plain cuda flavor.
cc_binary(
    name = "cuda-verbs/libplugin_UCX.so",
    linkopts = [
        "-Wl,-z,undefs",
        # Installed layout: <root>/lib/nixl/cuda-verbs/ → <root>/lib.
        "-Wl,-rpath,$$ORIGIN/../../../lib",
    ],
    linkshared = True,
    linkstatic = True,
    target_compatible_with = _LINUX_X86,
    deps = [
        ":ucx_plugin_lib_cuda_verbs",
        # Link against a real libibverbs.so.1 so the plugin's ibv_* undefined
        # symbols are recorded WITH version info (@IBVERBS_1.1 etc.). Left
        # unversioned (via -z undefs alone), the dynamic linker binds them to
        # the IBVERBS_1.0 compat definitions, whose struct ibv_device ABI
        # differs — device names read as garbage and UCX silently enumerates
        # zero RDMA devices. The DT_NEEDED this adds is the verbs flavor's
        # intended hard dependency on rdma-core.
        "@efa_libfabric_prebuilt//:libibverbs_import",
    ],
)

# CPU flavor of the UCX plugin, linked against the CUDA-free static UCX
# (tcp/shm/cma transports only). Unlike the CUDA flavor it has no load-time
# GPU driver dependencies, so it is dlopen-able on hosts with no GPU stack.
# The subdirectory in the target name gives the flavor its own directory:
# NIXL discovers plugins by the fixed filename `libplugin_UCX.so` within a
# single NIXL_PLUGIN_DIR, so consumers (e.g. the hermetic DRAM transfer tests
# on CPU-only CI workers) point NIXL_PLUGIN_DIR at the cpu/ directory.
cc_binary(
    name = "cpu/libplugin_UCX.so",
    linkopts = [
        "-Wl,-z,undefs",
    ],
    linkshared = True,
    linkstatic = True,
    target_compatible_with = _LINUX_X86,
    deps = [":ucx_plugin_lib_cpu"],
)

# CPU + verbs flavor: the cpu flavor plus the uct_ib RDMA transports, for a
# CPU-only process on an InfiniBand fabric — dKV, which runs on GPU hosts but
# registers host DRAM only and so must never pull in the CUDA stack. It carries
# the IB/mlx5 verbs transports only: the prebuilt's libuct_ib_efa.a is imported
# by no flavor, so this does not stand in for libplugin_LIBFABRIC_cpu.so on an
# EFA fabric, only on InfiniBand. It is the only UCX flavor the CPU-only
# nixl_prefix stages, so it needs no flavor subdirectory of its own at install
# time; the subdir here only keeps its filename distinct from cpu/.
cc_binary(
    name = "cpu-verbs/libplugin_UCX.so",
    linkopts = [
        "-Wl,-z,undefs",
        # Installed layout: <prefix>/lib/plugins/ → <prefix>/lib. The two
        # rdma-core libs below are DT_NEEDED, so without this the prefix is
        # self-contained only for a consumer that exports LD_LIBRARY_PATH.
        "-Wl,-rpath,$$ORIGIN/..",
    ],
    linkshared = True,
    linkstatic = True,
    target_compatible_with = _LINUX_X86,
    deps = [
        ":ucx_plugin_lib_cpu_verbs",
        # Link a real libibverbs.so.1 so the ibv_* undefined symbols are
        # recorded with version info; see cuda-verbs above for why leaving them
        # unversioned makes UCX silently enumerate zero RDMA devices.
        "@efa_libfabric_prebuilt//:libibverbs_import",
        # uct_ib_mlx5 calls the mlx5dv API. Linking it records a DT_NEEDED so
        # the loader resolves those symbols; NIXL dlopens plugins RTLD_NOW and
        # RTLD_LOCAL, so otherwise they would have to be preloaded RTLD_GLOBAL
        # by every process that creates an agent.
        "@efa_libfabric_prebuilt//:libmlx5_import",
    ],
)

# ROCm flavor for AMD-GPU hosts: carries the rocm_copy/rocm_ipc transports
# (libuct_rocm.a resolves against libhsa-runtime64.so.1 at load time). The
# non-verbs flavor keeps the plugin loadable on hosts without rdma-core
# (intranode transfers only); the rocm-verbs flavor below is preferred where
# rdma-core is present.
cc_binary(
    name = "rocm/libplugin_UCX.so",
    linkopts = [
        "-Wl,-z,undefs",
        # Installed layout: <root>/lib/nixl/rocm/ → <root>/lib.
        "-Wl,-rpath,$$ORIGIN/../../../lib",
    ],
    linkshared = True,
    linkstatic = True,
    target_compatible_with = _LINUX_X86,
    deps = [":ucx_plugin_lib_rocm"],
)

# ROCm + verbs flavor: a strict superset of the rocm flavor that adds the
# uct_ib RDMA transports for internode transfers (UCX picks transports per
# connection at runtime — same-node peers still use rocm_ipc/shm). It links
# libibverbs.so.1 alone, so its mlx5dv_* symbols rest on the RTLD_GLOBAL
# preload in _nixl_plugin_deps.py rather than on a DT_NEEDED; max._core
# selects this flavor only when both libibverbs.so.1 and libmlx5.so.1
# resolve, and otherwise falls back to the plain rocm flavor above.
cc_binary(
    name = "rocm-verbs/libplugin_UCX.so",
    linkopts = [
        "-Wl,-z,undefs",
        # Installed layout: <root>/lib/nixl/rocm-verbs/ → <root>/lib.
        "-Wl,-rpath,$$ORIGIN/../../../lib",
    ],
    linkshared = True,
    linkstatic = True,
    target_compatible_with = _LINUX_X86,
    deps = [
        ":ucx_plugin_lib_rocm_verbs",
        # Link against a real libibverbs.so.1 so the plugin's ibv_* undefined
        # symbols are recorded WITH version info (@IBVERBS_1.1 etc.). Left
        # unversioned (via -z undefs alone), the dynamic linker binds them to
        # the IBVERBS_1.0 compat definitions, whose struct ibv_device ABI
        # differs — device names read as garbage and UCX silently enumerates
        # zero RDMA devices. The DT_NEEDED this adds is the verbs flavor's
        # intended hard dependency on rdma-core.
        "@efa_libfabric_prebuilt//:libibverbs_import",
    ],
)

# In cuda/ because it is the CUDA-flavor libfabric build (EFA is an
# NVIDIA/AWS path; no ROCm libfabric exists) and it must sit in the same
# directory as the cuda UCX plugin for NIXL's single-dir discovery.
cc_binary(
    name = "cuda/libplugin_LIBFABRIC.so",
    linkopts = [
        "-Wl,-z,undefs",
    ],
    linkshared = True,
    linkstatic = True,
    target_compatible_with = _LINUX_X86,
    deps = [":libfabric_plugin_lib"],
)

# CPU-only build of the libfabric plugin, linked against the CUDA-free EFA
# libfabric prebuilt. Unconditional (no GPU-config select): a CPU-only consumer
# like dKV runs ON GPU hosts but must always get the CPU plugin + a matching
# CPU libfabric runtime — a config-dependent select would hand it the CUDA
# plugin (which needs a newer libfabric symbol version and the CUDA driver
# stack) on a GPU build host. Staged into the nixl_prefix as
# libplugin_LIBFABRIC.so (the name NIXL's loader discovers).
cc_binary(
    name = "libplugin_LIBFABRIC_cpu.so",
    linkopts = [
        "-Wl,-z,undefs",
        # Installed layout: <prefix>/lib/plugins/ → <prefix>/lib, where the
        # libfabric and rdma-core stack it needs is staged flat.
        "-Wl,-rpath,$$ORIGIN/..",
    ],
    linkshared = True,
    linkstatic = True,
    target_compatible_with = _LINUX_X86,
    deps = [":libfabric_plugin_lib_cpu"],
)

# Standalone shared objects for consumers that link NIXL dynamically (e.g. the
# nixl-sys Rust crate via NIXL_PREFIX) rather than folding the cc_library object
# code into their own .so. The `:nixl*` cc_libraries only yield object code; a
# real .so must be emitted, mirroring the plugin .so pattern and Sonny's libmixl.
#
# nixl-sys' NIXL_PREFIX path links -lnixl -lnixl_build -lnixl_common, matching
# upstream meson's library split: nixl=core (src/core), nixl_build=infra
# (src/infra: descriptors + memory_section), nixl_common=utils/common. As with
# the plugin .so targets above, each shared object folds in its alwayslink
# dependencies, so the symbol sets overlap; the dynamic linker interposes to the
# first definition (libnixl, linked first), which carries the complete set.
cc_binary(
    name = "libnixl.so",
    linkopts = [
        "-Wl,-z,undefs",
        # Set the SONAME so a downstream link records a clean `libnixl.so`
        # DT_NEEDED (rather than the bazel link-time path) and the runtime
        # loader keys off the SONAME.
        "-Wl,-soname,libnixl.so",
    ],
    linkshared = True,
    linkstatic = True,
    target_compatible_with = _LINUX_X86,
    deps = [":nixl"],
)

cc_binary(
    name = "libnixl_build.so",
    linkopts = [
        "-Wl,-z,undefs",
        "-Wl,-soname,libnixl_build.so",
    ],
    linkshared = True,
    linkstatic = True,
    target_compatible_with = _LINUX_X86,
    deps = [":nixl_infra"],
)

cc_binary(
    name = "libnixl_common.so",
    linkopts = [
        "-Wl,-z,undefs",
        "-Wl,-soname,libnixl_common.so",
    ],
    linkshared = True,
    linkstatic = True,
    target_compatible_with = _LINUX_X86,
    deps = [":nixl_common"],
)

# Dynamic-link import of libnixl.so for in-tree consumers (e.g. libmax) that
# want a DT_NEEDED on libnixl.so instead of folding the NIXL core object code
# into their own shared object. libnixl.so already folds the full transitive
# symbol set (core + infra + common + serdes + stream via alwayslink), so a
# single import resolves everything the `:nixl*` object libraries used to
# provide. Pair this with `:nixl_api_headers` for the compile-side includes.
cc_import(
    name = "nixl_shared",
    shared_library = ":libnixl.so",
    target_compatible_with = _LINUX_X86,
)

# Consumption wrapper for in-tree consumers (libmax, MLRT tests): NIXL public
# headers everywhere; on linux_x86_64 it also links libnixl.so (:nixl_shared)
# and carries the dlopen'd transport plugins as runfiles. Degrades to
# headers-only on macOS/aarch64 (the Linux-only targets enter the graph only via
# the linux_x86_64 select arm) so downstream targets like libmax still build
# there. This replaces the former //MLRT:Driver/NIXL wrapper.
cc_library(
    name = "nixl_runtime",
    data = select({
        "@@//:linux_x86_64": [
            ":cuda-verbs/libplugin_UCX.so",
            ":cuda/libplugin_LIBFABRIC.so",
            ":cuda/libplugin_UCX.so",
            ":rocm-verbs/libplugin_UCX.so",
            ":rocm/libplugin_UCX.so",
        ],
        "//conditions:default": [],
    }),
    deps = [":nixl_api_headers"] + select({
        "@@//:linux_x86_64": [":nixl_shared"],
        "//conditions:default": [],
    }),
)

# --- Install prefix -------------------------------------------------------
# Public C++ API headers in their api/cpp-relative layout. nixl.h includes its
# siblings as `#include "nixl_types.h"`, so a consumer points -I at
# <prefix>/include and uses `#include "nixl.h"`.
filegroup(
    name = "nixl_public_headers",
    srcs = glob(["src/api/cpp/**/*.h"]),
)

# Self-contained NIXL install-prefix tarball for non-Bazel consumers that link
# libnixl dynamically (e.g. the nixl-sys Rust crate via NIXL_PREFIX). Layout:
#   include/                              public headers (api/cpp tree)
#   lib/libnixl{,_build,_common}.so       the three shared objects nixl-sys links
#   lib/plugins/libplugin_LIBFABRIC.so    CPU libfabric backend plugin (EFA)
#   lib/plugins/libplugin_UCX.so          CPU UCX backend plugin, verbs (IB)
#   lib/<efa runtime>.so                  CPU EFA libfabric stack, flat in lib/
# The EFA runtime libs are bundled so the prefix is self-contained and Bazel
# materializes the prebuilt .so files even on a full remote-cache hit.
pkg_files(
    name = "nixl_prefix_headers",
    srcs = [":nixl_public_headers"],
    prefix = "include",
    strip_prefix = strip_prefix.from_pkg("src/api/cpp"),
)

pkg_files(
    name = "nixl_prefix_libs",
    srcs = [
        ":libnixl.so",
        ":libnixl_build.so",
        ":libnixl_common.so",
    ],
    prefix = "lib",
    strip_prefix = strip_prefix.files_only(),
)

# Always the CPU flavors (see libplugin_LIBFABRIC_cpu.so): the prefix is
# CPU-only regardless of the build host's GPU config. Both backends ship
# because the fabric is a property of the deployment, not of the build: consumers
# pick libfabric on EFA and UCX on InfiniBand at runtime. Renamed to the names
# NIXL's loader discovers (libplugin_<NAME>.so in a single flat directory).
#
# A second plugin here makes the consumer's choice load-bearing. NIXL returns
# discovered plugins from a std::set, so a consumer taking the first available
# one now gets LIBFABRIC by name ordering — right on EFA, and the one backend
# that cannot reach the fabric on InfiniBand. Deployments there must name the
# backend rather than rely on that ordering (CLIN-1730).
pkg_files(
    name = "nixl_prefix_plugin",
    srcs = [
        ":cpu-verbs/libplugin_UCX.so",
        ":libplugin_LIBFABRIC_cpu.so",
    ],
    prefix = "lib/plugins",
    renames = {":libplugin_LIBFABRIC_cpu.so": "libplugin_LIBFABRIC.so"},
    strip_prefix = strip_prefix.files_only(),
)

# CPU EFA libfabric runtime stack (libfabric.so.1, libefa, libibverbs, librdmacm,
# libnl, libnuma, libhwloc, libudev, and the EFA verbs provider), flattened into
# lib/ next to libnixl so the plugin's DT_NEEDED libs resolve via LD_LIBRARY_PATH.
pkg_files(
    name = "nixl_prefix_efa_libs",
    srcs = ["@efa_libfabric_prebuilt//:runtime_libs"],
    prefix = "lib",
    strip_prefix = strip_prefix.files_only(),
)

pkg_tar(
    name = "nixl_prefix",
    srcs = [
        ":nixl_prefix_efa_libs",
        ":nixl_prefix_headers",
        ":nixl_prefix_libs",
        ":nixl_prefix_plugin",
    ],
    out = "nixl-prefix.tar.gz",
    extension = "tar.gz",
    target_compatible_with = _LINUX_X86,
)
