"""Wrapper for the upstream cc_binary rule to apply modular specific defaults."""

load("@cc_compatibility_proxy//:proxy.bzl", _upstream_cc_binary = "cc_binary")
load("//bazel:config.bzl", "ALLOW_UNUSED_TAG")
load("//bazel/internal:config.bzl", "env_for_available_tools")  # buildifier: disable=bzl-visibility
load(":modular_clang_tidy_test.bzl", "modular_clang_tidy_test")
load(":mojo_test_environment.bzl", "mojo_test_environment")

_IGNORED_RUNTOOL_BINARIES = (
    "build-info",
    "compare-timings",
    "crash-report-path-info",
    "crash-test-dummy",
    "debugging_test_bin",
    "driver-tblgen",
    "gpu-query",
    "greeter-cli",
    "support-dialect-opt",
    "system-info",
)

def _cc_binary_impl(ctx):
    providers = ctx.super()

    output_group_info = None
    debug_package_info = None
    passthrough_providers = []
    for provider in providers:
        if type(provider) == "OutputGroupInfo":
            output_group_info = provider
        elif type(provider) == "struct" and hasattr(provider, "unstripped_file"):  # NOTE: Will require an update when this provider moves to starlark
            debug_package_info = provider
            passthrough_providers.append(provider)
        else:
            passthrough_providers.append(provider)

    if not output_group_info:
        fail("No OutputGroupInfo provider found")
    if not debug_package_info:
        fail("No DebugPackageInfo provider found")

    dsyms = getattr(output_group_info, "dsyms", depset())
    new_output_group_info = {}
    if dsyms:
        new_output_group_info["modular_dwarf"] = dsyms
    else:
        new_output_group_info["modular_dwarf"] = depset([debug_package_info.unstripped_file])

    for group in dir(output_group_info):
        new_output_group_info[group] = getattr(output_group_info, group)

    return passthrough_providers + [
        OutputGroupInfo(**new_output_group_info),
    ]

cc_binary = rule(
    implementation = _cc_binary_impl,
    parent = _upstream_cc_binary,
)

def modular_cc_binary(
        name,
        srcs = [],
        copts = [],
        data = [],
        deps = [],
        internal_deps = [],
        env = {},
        mojo_deps = [],
        toolchains = [],
        testonly = False,
        visibility = None,
        tags = [],
        **kwargs):
    """A wrapper for cc_binary

    https://bazel.build/reference/be/c-cpp#cc_binary

    Args:
        name: The name of the target
        srcs: See cc_library docs
        copts: See cc_library docs
        data: See cc_binary docs
        deps: See cc_binary docs
        internal_deps: Same as `deps`, but excluded for external builds.
        env: Env to set when using bazel run
        mojo_deps: mojo_library targets the binary depends on at runtime
        toolchains: See cc_binary docs
        testonly: See cc_binary docs
        visibility: See cc_binary docs
        tags: See cc_library docs
        **kwargs: Everything else passed through to cc_binary without processing
    """
    extra_deps = []
    extra_data = []
    extra_toolchains = []
    extra_env = env_for_available_tools(location_specifier = "execpath")
    if not testonly and name not in _IGNORED_RUNTOOL_BINARIES:
        extra_env |= {"RUNNING_DIRECTLY": "true"}
        extra_deps += select({
            "//:production_build": [],
            "//conditions:default": ["//bazel/internal:runtool"],
        })
    if mojo_deps:
        env_name = name + ".mojo_env"
        extra_toolchains.append(env_name)
        extra_data.append(env_name)
        extra_env |= {"MODULAR_MOJO_MAX_IMPORT_PATH": "$(COMPUTED_IMPORT_PATH)"}
        mojo_test_environment(
            name = env_name,
            data = mojo_deps,
            testonly = testonly,
            tags = ["manual"],
        )
    cc_binary(
        name = name,
        srcs = srcs,
        copts = copts,
        data = data + extra_data,
        deps = extra_deps + deps + internal_deps,
        env = extra_env | env,
        toolchains = toolchains + extra_toolchains,
        testonly = testonly,
        visibility = visibility,
        tags = tags,
        **kwargs
    )

    native.filegroup(
        name = name + ".dwarf",
        srcs = [":" + name],
        output_group = "modular_dwarf",
        testonly = testonly,
        tags = ["manual", ALLOW_UNUSED_TAG],
        visibility = visibility,
    )

    modular_clang_tidy_test(
        name = name,
        hdrs = [],
        srcs = srcs,
        copts = copts,
        tags = tags,
    )
