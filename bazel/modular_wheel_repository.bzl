"""A repository rule for creating wheel accessors. Not enabled by default for compatibility with modular's internal repo."""

load("@module_versions//:config.bzl", "PYTHON_VERSIONS_DOTTED")

_PLATFORM_MAPPINGS = {
    "linux_aarch64": "manylinux_2_34_aarch64",
    "linux_x86_64": "manylinux_2_34_x86_64",
    "macos_arm64": "macosx_13_0_arm64",
}

_WHEELS = [
    "max_core",
    "mojo_compiler",
]

def _rebuild_wheel(rctx):
    for py_version in PYTHON_VERSIONS_DOTTED:
        rctx.download_and_extract(
            url = "{base_url}/max/max-{version}-cp{py}-cp{py}-{platform}.whl".format(
                base_url = rctx.attr.base_url,
                version = rctx.attr.max_version,
                py = py_version.replace(".", ""),
                platform = _PLATFORM_MAPPINGS[rctx.attr.platform],
            ),
        )
    for name in _WHEELS:
        version = rctx.attr.mojo_version if name.startswith("mojo") else rctx.attr.max_version
        rctx.download_and_extract(
            url = "{}/{}/{}-{}-py3-none-{}.whl".format(
                rctx.attr.base_url,
                name.replace("_", "-"),
                name,
                version,
                _PLATFORM_MAPPINGS[rctx.attr.platform],
            ),
            strip_prefix = "{}-{}.data/platlib/".format(name, version),
        )

    # Platform-agnostic max mojo libs
    rctx.download_and_extract(
        url = "{base_url}/max-mojo-libs/max_mojo_libs-{version}-py3-none-any.whl".format(
            base_url = rctx.attr.base_url,
            version = rctx.attr.max_version,
        ),
        strip_prefix = "max_mojo_libs-{version}.data/platlib/".format(version = rctx.attr.max_version),
    )

    rctx.execute(["bash", "-c", "mv */platlib/max/_core.*.so max/"])
    rctx.execute(["mkdir", "-p", "max/_mlir/_mlir_libs"])
    rctx.execute(["bash", "-c", "mv */platlib/max/_mlir/_mlir_libs/_mlir.*.so max/_mlir/_mlir_libs/"])

    shared_lib_ext = "dylib" if rctx.attr.platform == "macos_arm64" else "so"
    rctx.file(
        "BUILD.bazel",
        # buildifier: disable=canonical-repository
        """
load("@rules_python//python:defs.bzl", "py_library")
load("@rules_cc//cc:defs.bzl", "cc_import")
load("@rules_mojo//mojo:mojo_import.bzl", "mojo_import")
load("@@//bazel:mojo_aliases.bzl", "INTERNAL_PACKAGES")

# Subdirectories of the wheel that are part of this repo and therefore should
# be removed so that they're not accidentally used when testing changes that
# depend on some closed-source portions of the wheel.
py_library(
    name = "max",
    data = glob([
        "max/_core.*",
        "max/_mlir/**",
        "modular/**",
    ], exclude = [
        "modular/lib/mojo/*",
    ]),
    pyi_srcs = glob([
        "max/**/*.pyi",
    ]),
    visibility = ["//visibility:public"],
    imports = ["."],
)

filegroup(
    name = "tblgen_python_srcs",
    srcs = [
        "max/_mlir/dialects/mo.py",
        "max/_mlir/dialects/rmo.py",
    ],
    visibility = ["//visibility:public"],
)

cc_import(
    name = "CompilerRT_lib",
    shared_library = glob(["modular/lib/libKGENCompilerRTShared.*"])[0],
    visibility = ["//visibility:public"],
)

INDIRECT_DEPENDENCIES = [
    "AsyncRTMojoBindings",
    "AsyncRTRuntimeGlobals",
    "KGENCompilerRTShared",
    "MGPRT",
    "MSupportGlobals",
]

[
    cc_import(
        name = "{}_lib".format(lib_name),
        shared_library = glob(["modular/lib/lib{}.*".format(lib_name)])[0],
        visibility = ["//visibility:public"],
    )
    for lib_name in INDIRECT_DEPENDENCIES
]

# Special case, NVPTX is platform-specific.
cc_import(
    name = "NVPTX_lib",
    shared_library = "modular/lib/libNVPTX.so",
    target_compatible_with = ["@platforms//os:linux"],
)

# libmax dynamically links libnixl.so, which ships in the wheel on
# linux_x86_64 only (NIXL is not built for aarch64 or macOS). Declared as a
# cc_import dep of max_lib (like the other indirect deps) so it is co-located
# with libmax in the solib tree and resolved at runtime.
cc_import(
    name = "nixl_lib",
    shared_library = "modular/lib/libnixl.so",
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
)

cc_import(
    name = "max_lib",
    shared_library = glob(["modular/lib/libmax.*"])[0],
    visibility = ["//visibility:public"],
    data = glob(["modular/lib/*.SHARED_LIB_EXT"]),
    deps = [":" + dep + "_lib" for dep in INDIRECT_DEPENDENCIES] + select({
        "@//:linux_x86_64": [":NVPTX_lib", ":nixl_lib"],
        "@//:linux_aarch64": [":NVPTX_lib"],
        "//conditions:default": [],
    })
)

[
    mojo_import(
        name = lib.split("/")[-1],
        mojodeps = ["modular/lib/mojo/" + lib.split("/")[-1] + ".mojoc"],
        visibility = ["//visibility:public"],
    )
    for lib in INTERNAL_PACKAGES
]
""".replace("SHARED_LIB_EXT", shared_lib_ext),
    )

rebuild_wheel = repository_rule(
    implementation = _rebuild_wheel,
    attrs = {
        "max_version": attr.string(
            mandatory = True,
        ),
        "mojo_version": attr.string(
            mandatory = True,
        ),
        "platform": attr.string(
            values = _PLATFORM_MAPPINGS.keys(),
            mandatory = True,
        ),
        "base_url": attr.string(
            mandatory = True,
        ),
    },
)

def _modular_wheel_repository_impl(rctx):
    rctx.file("BUILD.bazel", """
load("@rules_pycross//pycross:defs.bzl", "pycross_wheel_library")
load("@@//bazel:api.bzl", "requirement")
load("@@//bazel:mojo_aliases.bzl", "INTERNAL_PACKAGES")
load("@rules_python//python:defs.bzl", "py_binary")

alias(
    name = "wheel",
    actual = select({
        "@//:linux_aarch64": "@module_platlib_linux_aarch64//:max",
        "@//:linux_x86_64": "@module_platlib_linux_x86_64//:max",
        "@platforms//os:macos": "@module_platlib_macos_arm64//:max",
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "tblgen_python_srcs",
    actual = select({
        "@//:linux_aarch64": "@module_platlib_linux_aarch64//:tblgen_python_srcs",
        "@//:linux_x86_64": "@module_platlib_linux_x86_64//:tblgen_python_srcs",
        "@platforms//os:macos": "@module_platlib_macos_arm64//:tblgen_python_srcs",
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "max_lib",
    actual = select({
        "@//:linux_aarch64": "@module_platlib_linux_aarch64//:max_lib",
        "@//:linux_x86_64": "@module_platlib_linux_x86_64//:max_lib",
        "@platforms//os:macos": "@module_platlib_macos_arm64//:max_lib",
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "AsyncRTMojoBindings_lib",
    actual = select({
        "@//:linux_aarch64": "@module_platlib_linux_aarch64//:AsyncRTMojoBindings_lib",
        "@//:linux_x86_64": "@module_platlib_linux_x86_64//:AsyncRTMojoBindings_lib",
        "@platforms//os:macos": "@module_platlib_macos_arm64//:AsyncRTMojoBindings_lib",
    }),
    visibility = ["//visibility:public"],
)

alias(
    name = "CompilerRT_lib",
    actual = select({
        "@//:linux_aarch64": "@module_platlib_linux_aarch64//:CompilerRT_lib",
        "@//:linux_x86_64": "@module_platlib_linux_x86_64//:CompilerRT_lib",
        "@platforms//os:macos": "@module_platlib_macos_arm64//:CompilerRT_lib",
    }),
    visibility = ["//visibility:public"],
)

[
    alias(
        name = lib.split("/")[-1],
        actual = select({
            "@//:linux_aarch64": "@module_platlib_linux_aarch64//:" + lib.split("/")[-1],
            "@//:linux_x86_64": "@module_platlib_linux_x86_64//:" + lib.split("/")[-1],
            "@platforms//os:macos": "@module_platlib_macos_arm64//:" + lib.split("/")[-1],
        }),
        visibility = ["//visibility:public"],
    )
    for lib in INTERNAL_PACKAGES
]
""")  # buildifier: disable=canonical-repository

modular_wheel_repository = repository_rule(
    implementation = _modular_wheel_repository_impl,
)
