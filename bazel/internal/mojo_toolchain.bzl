"""An internal Mojo toolchain to point to our local tools."""

load("@cfg_workaround.bzl", "CFG_WORKAROUND")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_mojo//mojo:providers.bzl", "MojoInfo", "MojoToolchainInfo")

MojoBuildEnvInfo = provider(
    doc = "Extra environment variables to merge into Mojo build-tool action envs.",
    fields = {
        "env": "dict[str, str] merged into the env of actions that invoke mojo as a build tool",
    },
)

def _mojo_toolchain_impl(ctx):
    tool_files = []
    for dep in [ctx.attr.lld, ctx.attr.mojo]:
        tool_files.append(dep[DefaultInfo].default_runfiles.files)
        tool_files.append(dep[DefaultInfo].files)

    copts = []
    gpu_toolchain = ctx.toolchains["@rules_mojo//:gpu_toolchain_type"]
    if gpu_toolchain:
        copts.append("--target-accelerator=" + gpu_toolchain.mojo_gpu_toolchain_info.target_accelerator)

    is_macos = ctx.target_platform_has_constraint(ctx.attr._macos_constraint[platform_common.ConstraintValueInfo])
    if is_macos:
        min_os = ctx.fragments.cpp.minimum_os_version() or ctx.fragments.apple.macos_minimum_os_flag
        if min_os:
            copts.append("--target-triple=arm64-apple-macosx{}".format(min_os))

    copts_toolchain = ctx.toolchains["@rules_mojo//:copts_toolchain_type"]
    if copts_toolchain:
        copts.extend(copts_toolchain.copts_toolchain_info.copts)
        precompile_copts = copts_toolchain.copts_toolchain_info.precompile_copts
    else:
        precompile_copts = []

    # Expose compiler plugin env for build actions.  Rules that invoke the Mojo
    # compiler (kgen.bzl, and rules_mojo via patch) merge this into their action
    # env so that MODULAR_COMPILER_PLUGINS is set automatically when the plugin
    # is present, without requiring a hardcoded --action_env in local.bazelrc.
    build_env = {}
    if ctx.file.compiler_plugin:
        build_env["MODULAR_COMPILER_PLUGINS"] = ctx.file.compiler_plugin.path
        tool_files.append(depset([ctx.file.compiler_plugin]))

    # Let a plugin contribute extra build-action env (e.g. a plugin-specific SDK
    # lib path) so it reaches only mojo build actions, not a global --action_env
    # that would pollute every action's cache key.
    if ctx.attr.extra_build_env:
        build_env.update(ctx.attr.extra_build_env[MojoBuildEnvInfo].env)

    # Expose driver plugin for test runtime.  The runtime_env and
    # extra_runfiles fields are merged by rules_mojo into RunEnvironmentInfo
    # so tests find the plugin without a hardcoded --test_env in local.bazelrc.
    runtime_env = {}
    extra_runfiles = []
    if ctx.file.driver_plugin:
        runtime_env["MODULAR_DRIVER_PLUGINS"] = ctx.file.driver_plugin.short_path
        extra_runfiles.append(ctx.file.driver_plugin)

    return [
        platform_common.ToolchainInfo(
            mojo_toolchain_info = MojoToolchainInfo(
                all_tools = tool_files,
                copts = copts,
                precompile_copts = precompile_copts,
                lld = ctx.executable.lld,
                mojo = ctx.executable.mojo,
                implicit_deps = ctx.attr.implicit_deps,
            ),
            build_env = build_env,
            runtime_env = runtime_env,
            extra_runfiles = extra_runfiles,
        ),
    ]

mojo_toolchain = rule(
    implementation = _mojo_toolchain_impl,
    attrs = {
        "lld": attr.label(
            allow_files = True,
            mandatory = True,
            executable = True,
            cfg = CFG_WORKAROUND,  # NOTE: This differs from the rules_mojo toolchain
            doc = "The lld executable to link with.",
        ),
        "mojo": attr.label(
            mandatory = True,
            executable = True,
            cfg = CFG_WORKAROUND,  # NOTE: This differs from the rules_mojo toolchain
            doc = "The mojo compiler executable to build with.",
        ),
        "implicit_deps": attr.label_list(
            providers = [[CcInfo], [MojoInfo]],
            mandatory = True,
            cfg = "target",
            doc = "Implicit dependencies that every target should depend on, providing either CcInfo, or MojoInfo.",
        ),
        "compiler_plugin": attr.label(
            mandatory = False,
            allow_single_file = True,
            cfg = CFG_WORKAROUND,
            doc = "Optional compiler plugin (.so) to set via MODULAR_COMPILER_PLUGINS in build actions.",
        ),
        "driver_plugin": attr.label(
            mandatory = False,
            allow_single_file = True,
            cfg = "target",
            doc = "Optional driver plugin (.so) to set via MODULAR_DRIVER_PLUGINS at test runtime.",
        ),
        "extra_build_env": attr.label(
            mandatory = False,
            providers = [MojoBuildEnvInfo],
            cfg = "target",
            doc = "Optional target providing MojoBuildEnvInfo with extra build-action env vars.",
        ),
        "_macos_constraint": attr.label(
            default = Label("@platforms//os:macos"),
        ),
    },
    toolchains = [
        config_common.toolchain_type("@rules_mojo//:copts_toolchain_type", mandatory = False),
        config_common.toolchain_type("@rules_mojo//:gpu_toolchain_type", mandatory = False),
    ],
    fragments = ["cpp", "apple"],
)

def _mojo_build_env_impl(ctx):
    return [MojoBuildEnvInfo(env = ctx.attr.env)]

mojo_build_env = rule(
    implementation = _mojo_build_env_impl,
    attrs = {
        "env": attr.string_dict(
            doc = "Environment variables to merge into Mojo build-tool actions.",
        ),
    },
    doc = "Vends extra build-action env vars to mojo_toolchain via MojoBuildEnvInfo.",
)
