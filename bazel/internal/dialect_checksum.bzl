"""Bazel rule to compute a SHA256 checksum of MLIR dialect .td files."""

load("@cfg_workaround.bzl", "CFG_WORKAROUND")

def _dialect_checksum_impl(ctx):
    output = ctx.actions.declare_file(ctx.attr.out)

    # Collect all .td files from the srcs (which are td_library targets).
    td_files = []
    for src in ctx.attr.srcs:
        td_files.extend(src.files.to_list())

    args = ctx.actions.args()
    args.add("-o", output)
    args.add_all(sorted(td_files, key = lambda f: f.path))

    ctx.actions.run(
        outputs = [output],
        inputs = td_files,
        executable = ctx.executable._tool,
        arguments = [args],
        execution_requirements = {"supports-path-mapping": "1"},
        progress_message = "Computing dialect checksum %{label}",
    )

    return [DefaultInfo(files = depset([output]))]

dialect_checksum = rule(
    implementation = _dialect_checksum_impl,
    attrs = {
        "srcs": attr.label_list(
            doc = "td_library targets whose .td files should be checksummed.",
        ),
        "out": attr.string(
            default = "GeneratedDialectChecksum.h",
            doc = "Output header filename.",
        ),
        "_tool": attr.label(
            default = Label("//bazel/internal:gen_dialect_checksum"),
            cfg = CFG_WORKAROUND,
            executable = True,
        ),
    },
)
