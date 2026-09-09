load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "ibverbs_provider_files",
    srcs = [
        "etc/libibverbs.d/efa.driver",
        "etc/libibverbs.d/mlx5.driver",
        "lib/libibverbs/libefa-rdmav59.so",
        "lib/libibverbs/libmlx5-rdmav59.so",
    ],
)

cc_import(
    name = "libefa_import",
    shared_library = "lib/libefa.so.1",
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)

cc_import(
    name = "libfabric_import",
    shared_library = "lib/libfabric.so.1",
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)

cc_import(
    name = "libhwloc_import",
    shared_library = "lib/libhwloc.so.15",
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)

cc_import(
    name = "libnuma_import",
    shared_library = "lib/libnuma.so.1",
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)

cc_import(
    name = "libibverbs_import",
    shared_library = "lib/libibverbs.so.1",
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)

# The mlx5dv API the UCX verbs transports call. Its provider sibling
# (lib/libibverbs/libmlx5-rdmav59.so) is loaded by libibverbs at runtime and so
# is not imported here.
cc_import(
    name = "libmlx5_import",
    shared_library = "lib/libmlx5.so.1",
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)

cc_import(
    name = "librdmacm_import",
    shared_library = "lib/librdmacm.so.1",
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)

cc_import(
    name = "libnl_3_import",
    shared_library = "lib/libnl-3.so.200",
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)

cc_import(
    name = "libnl_route_3_import",
    shared_library = "lib/libnl-route-3.so.200",
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)

cc_import(
    name = "libudev_import",
    shared_library = "lib/libudev.so.1",
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)

cc_library(
    name = "hwloc",
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    deps = [
        ":libhwloc_import",
        ":libudev_import",
    ],
)

cc_library(
    name = "numa",
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    deps = [":libnuma_import"],
)

cc_library(
    name = "libfabric",
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    deps = [
        ":libefa_import",
        ":libfabric_import",
        ":libibverbs_import",
        ":libnl_3_import",
        ":libnl_route_3_import",
        ":librdmacm_import",
    ],
)

# The full set of prebuilt runtime shared objects (libfabric + the rdma-core
# stack + the EFA verbs provider), as plain files. cc_import exposes these only
# through CcInfo, so a non-cc consumer that needs to ship them (e.g. a genrule
# assembling a self-contained install prefix) references this filegroup. Using
# it as a genrule input also forces Bazel to materialize the .so files locally
# even when the build is otherwise served entirely from the remote cache.
filegroup(
    name = "runtime_libs",
    srcs = glob([
        "lib/*.so*",
        "lib/libibverbs/*.so*",
    ]),
)
