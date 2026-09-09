"""Common tablegen abstraction"""

load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//bazel:config.bzl", "ALLOW_UNUSED_TAG")

def driver_option_tablegen(
        name,
        srcs,
        td_file,
        deps = [],
        includes = [],
        opts = [],
        manual_name = None,
        markdown_name = None,
        include_prefix = None,
        strip_include_prefix = None):
    """Run tablegen for driver options.

    Run tablegen for driver options.

    Args:
        name: The prefix of the underlying targets
        srcs: All sources relevant to processing the file
        td_file: The tablegen file we're processing
        deps: Any tablegen deps required to process the file
        includes: cc_library includes
        opts: Extra tablegen opts to pass to every tablegen invocation
        manual_name: A custom name for the output man page
        markdown_name: A custom name for the output markdown page
        include_prefix: A custom include_prefix for the cc_library
        strip_include_prefix: A custom strip_include_prefix for the cc_library
    """
    library_name = name + ".td_library"
    td_library(
        name = library_name,
        includes = includes,
        srcs = srcs,
        deps = deps,
    )

    name_without_extension = td_file.split(".")[0]

    # FIXME: wrong path when overrides are used
    manual_output = (manual_name or name_without_extension) + ".1"
    markdown_output = (markdown_name or name_without_extension) + ".md"

    gentbl_cc_library(
        name = name + ".inc_gen",
        tbl_outs = [(
            ["-gen-opt-parser-defs"] + opts,
            name_without_extension + ".inc",
        )],
        tblgen = "@llvm-project//llvm:llvm-tblgen",
        includes = includes,
        td_file = td_file,
        deps = deps + [library_name],

        # TODO: These shouldn't all have this but this expands to another macro
        # where some underlying targets are unused
        tags = [ALLOW_UNUSED_TAG],
    )

    gentbl_cc_library(
        name = name + ".help_gen",
        tbl_outs = [
            (
                ["-gen-help-text"] + opts,
                name_without_extension + "HelpText.inc",
            ),
            (
                ["-gen-help-hidden-text"] + opts,
                name_without_extension + "HelpHiddenText.inc",
            ),
            (["-gen-man-page"] + opts, manual_output),
            (["-gen-markdown"] + opts, markdown_output),
        ],
        tblgen = "//Support/tools/driver-tblgen",
        td_file = td_file,
        deps = deps + [library_name],
        tags = [ALLOW_UNUSED_TAG],
    )

    native.filegroup(
        name = name + ".all_files",
        srcs = [
            manual_output,
            markdown_output,
            name_without_extension + ".inc",
            name_without_extension + "HelpText.inc",
            name_without_extension + "HelpHiddenText.inc",
        ],
        tags = [ALLOW_UNUSED_TAG],
    )
    cc_library(
        name = name + ".inc_files",
        hdrs = [
            name_without_extension + ".inc",
            name_without_extension + "HelpText.inc",
            name_without_extension + "HelpHiddenText.inc",
        ],
        include_prefix = include_prefix,
        strip_include_prefix = strip_include_prefix,
        tags = [ALLOW_UNUSED_TAG],
    )

    native.filegroup(
        name = name + ".man_files",
        srcs = [
            manual_output,
        ],
        tags = [ALLOW_UNUSED_TAG],
    )

    native.filegroup(
        name = name + ".md_files",
        srcs = [
            markdown_output,
        ],
        tags = [ALLOW_UNUSED_TAG],
    )

def gentbl_mlir_dialect(
        name,
        dialect,
        td_file,
        extra_tbl_outs = [],
        deps = []):
    """Create all tablegen files for a mlir dialect

    This mirrors add_mlir_dialect from cmake

    Args:
        name: The name of the underlying cc_library that's generated
        dialect: The name of the dialect
        td_file: The tablegen file to base generation on
        extra_tbl_outs: Extra tablegen invocations to register besides the defaults
        deps: Any tablegen file deps required
    """
    name_without_extension = td_file.split(".")[0]
    gentbl_cc_library(
        name = name,
        includes = ["include"],
        tbl_outs = [
            (["-gen-op-decls"], name_without_extension + ".h.inc"),
            (["-gen-op-defs"], name_without_extension + ".cpp.inc"),
            (
                ["-gen-typedef-decls", "-typedefs-dialect=" + dialect],
                name_without_extension + "Types.h.inc",
            ),
            (
                ["-gen-typedef-defs", "-typedefs-dialect=" + dialect],
                name_without_extension + "Types.cpp.inc",
            ),
            (
                ["-gen-dialect-decls", "-dialect=" + dialect],
                name_without_extension + "Dialect.h.inc",
            ),
            (
                ["-gen-dialect-defs", "-dialect=" + dialect],
                name_without_extension + "Dialect.cpp.inc",
            ),
        ] + extra_tbl_outs,
        tblgen = "@llvm-project//mlir:mlir-tblgen",
        td_file = td_file,
        deps = deps,
    )

    gentbl_filegroup(
        name = dialect.upper() + "Dialect.doc",
        includes = ["include"],
        tbl_outs = [(["-gen-dialect-doc", "-dialect=" + dialect], td_file.removesuffix(".td") + "Dialect.md")],
        tblgen = "@llvm-project//mlir:mlir-tblgen",
        td_file = td_file,
        deps = deps,
    )

def gentbl_modular_passes(name, td_file, output_name = None, deps = []):
    """Add a tablegen target for the common passes pattern

    Args:
        name: The target name
        td_file: The tablegen file to process
        output_name: Custom name for the output, passed as the -name arg
        deps: tablegen file dependencies of the target
    """
    args = ["-gen-pass-decls"]
    if output_name:
        args.extend(["-name", output_name])

    td_name = td_file.split("/")[-1].split(".")[0]
    output_name = output_name or td_name
    output_path = td_file.rsplit("/", 1)[0] + "/" + output_name + ".h.inc"

    gentbl_cc_library(
        name = name,
        includes = ["include"],
        tbl_outs = [(args, output_path)],
        tblgen = "@llvm-project//mlir:mlir-tblgen",
        td_file = td_file,
        deps = deps,
    )

    gentbl_filegroup(
        name = name + ".doc",
        includes = ["include"],
        tbl_outs = [(["-gen-pass-doc"], td_file.rsplit("/", 1)[0] + "/" + output_name + ".md")],
        tblgen = "@llvm-project//mlir:mlir-tblgen",
        td_file = td_file,
        deps = deps,
        tags = [ALLOW_UNUSED_TAG],
    )
