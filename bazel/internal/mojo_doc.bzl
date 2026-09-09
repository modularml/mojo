"""Generate documentation for Mojo source files."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("@rules_mojo//mojo:providers.bzl", "MojoInfo")
load("@rules_mojo//mojo/private:utils.bzl", "MOJO_EXTENSIONS", "collect_mojoinfo", "format_import")  # buildifier: disable=bzl-visibility

def _format_include(file):
    return ["-I", file.dirname]

def _root_directory(file):
    # Pass srcs[0]'s dir through map_each so Bazel path-mapping rewrites it. A
    # raw `.dirname` string is left unmapped, so a generated srcs[0] (e.g. an
    # overlaid stdlib tree) would point at the un-stripped bazel-out path where
    # the path-mapped inputs were never materialized.
    return file.dirname

def _mojo_doc_implementation(ctx):
    mojo_toolchain = ctx.toolchains["@rules_mojo//:toolchain_type"].mojo_toolchain_info

    import_paths, transitive_mojodeps, _ = collect_mojoinfo(ctx.attr.deps + mojo_toolchain.implicit_deps)
    root_directory = ctx.files.srcs[0].dirname

    file_args = ctx.actions.args()
    for file in ctx.files.srcs:
        if not file.dirname.startswith(root_directory):
            file_args.add_all([file], map_each = _format_include)

    file_args.add_all(import_paths, map_each = format_import)
    file_args.add_all([ctx.files.srcs[0]], map_each = _root_directory)

    mojodoc_output = ctx.actions.declare_file(ctx.label.name + ".mojodoc.json")
    ctx.actions.run(
        executable = mojo_toolchain.mojo,
        inputs = depset(ctx.files.srcs, transitive = [transitive_mojodeps]),
        tools = mojo_toolchain.all_tools,
        outputs = [mojodoc_output],
        mnemonic = "MojoDoc",
        arguments = [
                        "doc",
                        "-Werror",
                        "-o",
                        mojodoc_output.path,
                        file_args,
                    ] + (["--docs-base-path", ctx.attr.docs_base_path] if ctx.attr.docs_base_path else []) +
                    (["--diagnose-missing-doc-strings"] if ctx.attr.validate_missing_docs else []) +
                    ctx.attr.copts,
        progress_message = "%{label} generating mojodoc.json",
        env = {
            "MODULAR_HOME": ".",  # Make sure any cache files are written to somewhere bazel will cleanup
        },
        use_default_shell_env = True,
    )

    doc_output = ctx.actions.declare_directory(ctx.label.name)
    markdown_args = ctx.actions.args()
    markdown_args.add("--show-stability-markers", ctx.attr.show_stability_markers)
    markdown_args.add("-o", doc_output.path)
    markdown_args.add(mojodoc_output.path)
    if ctx.attr.docs_title:
        markdown_args.add("--docs-title", ctx.attr.docs_title)
    if ctx.attr.stability_doc_url:
        markdown_args.add("--stability-doc-url", ctx.attr.stability_doc_url)
    if ctx.attr.docs_hosted_on_mojolang:
        markdown_args.add("--hosted-on-mojolang")

    ctx.actions.run(
        executable = ctx.executable._mojodoc_json_to_markdown,
        outputs = [doc_output],
        inputs = [mojodoc_output],
        mnemonic = "MojoDoc",
        arguments = [markdown_args],
        use_default_shell_env = True,
    )

    return [
        DefaultInfo(files = depset([doc_output])),
    ]

mojo_doc = rule(
    implementation = _mojo_doc_implementation,
    attrs = {
        "srcs": attr.label_list(
            allow_empty = False,
            allow_files = MOJO_EXTENSIONS,
        ),
        "deps": attr.label_list(
            # Matches mojo_library, whose deps this mirrors: cc deps are
            # ignored when generating docs but must not be rejected.
            providers = [[MojoInfo], [CcInfo]],
        ),
        "validate_missing_docs": attr.bool(
            default = False,
            doc = "Fail for missing docstrings",
        ),
        "docs_base_path": attr.string(
            default = "",
            doc = "Base path prefix for generated documentation links",
        ),
        "docs_title": attr.string(
            default = "",
            doc = "Custom title for the top-level package index page",
        ),
        "docs_hosted_on_mojolang": attr.bool(
            default = False,
            doc = (
                "True when generated Markdown is published on mojolang.org (stdlib, layout). " +
                "Passes ``--hosted-on-mojolang`` to ``mojodoc_json_to_markdown`` so std/layout " +
                "links stay root-relative; kernel docs on Modular omit this."
            ),
        ),
        "show_stability_markers": attr.string(
            default = "none",
            values = ["all", "stable", "none"],
            doc = "Show stability markers in generated docs",
        ),
        "stability_doc_url": attr.string(
            default = "",
            doc = "URL that stability markers link to. Markers are plain text when unset.",
        ),
        "copts": attr.string_list(
            default = [],
            doc = "Extra flags passed to `mojo doc` (e.g. `--ignore-deprecated=...`). " +
                  "Unlike `mojo_library`'s `copts`, these are not compiler flags in " +
                  "the general sense -- only flags `mojo doc`'s own option table " +
                  "recognizes are valid here.",
        ),
        "_mojodoc_json_to_markdown": attr.label(
            default = Label("//bazel/internal:mojodoc_json_to_markdown"),
            cfg = "exec",
            executable = True,
        ),
    },
    toolchains = ["@rules_mojo//:toolchain_type"],
)
