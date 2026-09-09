load("@rules_cc//cc:cc_import.bzl", "cc_import")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_import(
    name = "tcmalloc_lib",
    static_library = "libtcmalloc_minimal.a",
    target_compatible_with = select({
        "@platforms//os:macos": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tcmalloc",
    hdrs = glob(["gperftools/*.h"]),
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":tcmalloc_lib",
    ],
)
