# Bazel overlay for the hack-to-owned xgrammar grammar engine
# (github.com/mlc-ai/xgrammar, v0.2.3): builds only the torch/tvm-ffi-free C++
# grammar core (cpp/ minus cpp/tvm_ffi/) into one cc_library. DLPack comes from
# @dlpack -- it's a git submodule absent from the release tarball; picojson ships
# in the tarball.

load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "xgrammar",
    srcs = glob(
        ["cpp/**/*.cc"],
        exclude = ["cpp/tvm_ffi/**"],
    ),
    hdrs = glob(
        [
            "include/xgrammar/**/*.h",
            "cpp/**/*.h",
        ],
        exclude = ["cpp/tvm_ffi/**"],
    ) + ["3rdparty/picojson/picojson.h"],
    # MAX defaults to -fno-exceptions/-fno-rtti, but the grammar core needs both
    # (support/logging.h throws, reflection.h uses RTTI); -w silences upstream
    # warnings.
    copts = [
        "-w",
        "-fexceptions",
        "-frtti",
    ],
    includes = [
        "3rdparty/picojson",
        "cpp",
        "include",
    ],
    local_defines = ["XGRAMMAR_ENABLE_CPPTRACE=0"],
    deps = ["@dlpack"],
)
