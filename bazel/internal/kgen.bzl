"""Rules specific to KGEN"""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@cfg_workaround.bzl", "CFG_WORKAROUND")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_mojo//mojo:providers.bzl", "MojoInfo")

def _sanitizer_args(ctx):
    config = ctx.attr._modular_default_constraint[BuildSettingInfo].value
    if config == "asan":
        return ["--sanitize", "address"]

    return []

def _build_type_args(ctx):
    config = ctx.attr._modular_default_constraint[BuildSettingInfo].value
    if config in ("debug_modular", "debug_everything"):
        debug_level = "full"
        optimization_level = 0
    elif config == "dev":
        debug_level = "full"
        optimization_level = 2
    elif config in ("default", "asan", "tsan", "ubsan"):
        debug_level = "full"
        optimization_level = 3
    elif config in ("ci_build", "release", "production"):
        debug_level = "none"
        optimization_level = 3
    else:
        fail("bazel misconfiguration")

    return [
        "-O{}".format(optimization_level),
        "--debug-level={}".format(debug_level),
    ]

def _format_include(arg):
    return ["-I", arg.dirname]

def _kgen_kernel_impl(ctx):
    tc = ctx.toolchains["@rules_mojo//:toolchain_type"]
    mojo_toolchain = tc.mojo_toolchain_info

    output_name = ctx.file.src.basename.split(".")[0]
    header = ctx.actions.declare_file(output_name + ".h")
    archive = ctx.actions.declare_file(output_name + ".a")

    mojodeps = []
    import_args = ctx.actions.args()
    for dep in ctx.attr.deps:
        if MojoInfo in dep:
            mojodeps.append(dep[MojoInfo].mojodeps)
            import_args.add_all(dep[MojoInfo].mojodeps, map_each = _format_include)

    shared_args = {
        "executable": ctx.executable._kgen,
        "inputs": depset([ctx.file.src] + ctx.files.data, transitive = mojodeps),
        "use_default_shell_env": True,
        "env": {
            "MODULAR_MOJO_MAX_LLD_PATH": "/dev/null",
            "MODULAR_HOME": ".",
            "ZERO_AR_DATE": "1",
        } | getattr(tc, "build_env", {}),
        "progress_message": "%{label} generating %{output}",
        "execution_requirements": {
            "supports-path-mapping": "1",
        },
    }

    # TODO: Fix -W* exclusion hack
    args = [import_args] + [x for x in mojo_toolchain.copts if x not in ("-Werror", "-Wno-error")]
    args.extend(_sanitizer_args(ctx))
    args.extend(_build_type_args(ctx))

    header_args = ctx.actions.args()
    header_args.add("-emit=header")
    header_args.add("-o", header)
    header_args.add(ctx.file.src)
    ctx.actions.run(
        outputs = [header],
        arguments = [header_args] + args,
        **shared_args
    )

    archive_args = ctx.actions.args()
    archive_args.add("-emit=object")
    archive_args.add("-o", archive)
    archive_args.add(ctx.file.src)
    ctx.actions.run(
        outputs = [archive],
        arguments = [archive_args] + args,
        **shared_args
    )

    return [
        DefaultInfo(files = depset([header, archive])),
        cc_common.merge_cc_infos(
            direct_cc_infos = [
                CcInfo(
                    compilation_context = cc_common.create_compilation_context(
                        headers = depset([header]),
                        includes = depset([header.dirname]),
                    ),
                    linking_context = cc_common.create_linking_context(
                        linker_inputs = depset([
                            cc_common.create_linker_input(
                                owner = ctx.label,
                                libraries = depset([
                                    cc_common.create_library_to_link(
                                        actions = ctx.actions,
                                        pic_static_library = archive,
                                    ),
                                ]),
                            ),
                        ]),
                    ),
                ),
            ],
            cc_infos = [ctx.attr._compiler_rt[CcInfo]],
        ),
    ]

kgen_kernel = rule(
    implementation = _kgen_kernel_impl,
    attrs = {
        "src": attr.label(allow_single_file = True, mandatory = True),
        "data": attr.label_list(allow_files = True),
        "deps": attr.label_list(providers = [MojoInfo]),
        "_compiler_rt": attr.label(
            default = Label("//Mojo:CompilerRT"),
            cfg = "target",
            providers = [CcInfo],
        ),
        "_kgen": attr.label(
            default = Label("//Mojo/tools/kgen"),
            cfg = CFG_WORKAROUND,
            executable = True,
        ),
        "_modular_default_constraint": attr.label(
            default = Label("//:modular_config"),
        ),
    },
    toolchains = [
        "@rules_mojo//:toolchain_type",
    ],
    fragments = ["cpp"],
)
