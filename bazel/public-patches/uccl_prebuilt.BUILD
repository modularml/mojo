# Defines the imported pre-built UCCL transport library (libuccl_p2p.so) and
# the headers the NIXL UCCL backend plugin compiles against. Produced by
# utils/build-uccl.sh and published as the @uccl_prebuilt bazel artifact.

load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# UCCL's transport library. It has undefined Python interpreter symbols
# (PyGILState_* etc.) — an upstream limitation (it presumes a Python host) —
# so consumers link it with `-z undefs` and it resolves those at load time in
# a Python process.
cc_import(
    name = "libuccl_p2p_import",
    shared_library = "lib/libuccl_p2p.so",
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)

# The bare .so, for staging alongside libplugin_UCCL.so in the packaged
# lib/nixl/rocm-uccl/ dir (the plugin resolves it via its $ORIGIN rpath).
filegroup(
    name = "libuccl_p2p_so",
    srcs = ["lib/libuccl_p2p.so"],
)

# Top-level target consumers depend on: the imported .so plus the p2p headers
# the plugin includes (`uccl_engine.h`, which pulls in `common.h` and the
# other util headers).
cc_library(
    name = "uccl",
    hdrs = glob(["include/**/*.h"]),
    includes = [
        # `include` for <infiniband/*.h> (rdma-core headers UCCL's common.h
        # pulls in); the p2p dirs for "uccl_engine.h" and its "common.h".
        "include",
        "include/p2p",
        "include/p2p/util",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    deps = [":libuccl_p2p_import"],
)
