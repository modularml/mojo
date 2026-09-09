"""A genrule variant that avoids the exec transition to reduce duplicate building"""

load("@cfg_workaround.bzl", "CFG_WORKAROUND")

def _modular_genrule_impl(ctx):
    tools = []
    for target in ctx.attr.tools + ctx.attr.srcs:
        tools.append(target[DefaultInfo].default_runfiles.files)
        tools.append(target[DefaultInfo].files_to_run)

    # https://github.com/bazelbuild/bazel/issues/23200
    inputs, command, _ = ctx.resolve_command(
        command = ctx.attr.cmd,
        attribute = "cmd",
        expand_locations = True,
        tools = ctx.attr.tools + ctx.attr.srcs,
    )

    ctx.actions.run_shell(
        outputs = ctx.outputs.outs,
        tools = tools,
        inputs = inputs,
        command = command[-1].replace("$$", "$"),
        use_default_shell_env = True,
        env = {"MODULAR_HOME": "."},
    )

modular_genrule = rule(
    implementation = _modular_genrule_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "outs": attr.output_list(mandatory = True),
        "cmd": attr.string(mandatory = True),
        "tools": attr.label_list(
            cfg = CFG_WORKAROUND,  # NOTE: This is the primary difference from genrule
        ),
    },
)
