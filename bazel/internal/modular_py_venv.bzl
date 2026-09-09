"""Create a virtualenv including the dependencies of the given targets."""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:py_info.bzl", "PyInfo")

def _collect_venv_files_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".json")

    transitive_imports = []
    external_python_files = set()
    data_files = set()

    # NOTE: It's a bit weird that we fetch PyInfo from data, but we do some weird stuff with binaries as data
    for dep in ctx.attr.deps + ctx.attr.data:
        if PyInfo in dep:
            transitive_imports.append(dep[PyInfo].imports)

            # Skip symlinking internal files, they will be picked up by imports from the runfiles instead.
            external_python_files |= set([x.short_path for x in dep[PyInfo].transitive_sources.to_list() if x.short_path.startswith("../")])

        for file in dep[DefaultInfo].default_runfiles.files.to_list():
            # Only collect binaries, shared libraries (including versioned .so.*), and Mojo precompiled files.
            if not file.extension in ("", "so", "dylib", "mojoc") or ".so." in file.basename:
                continue

            # Directories only matter if they have files in them.
            if file.is_directory:
                continue

            # TODO: Fix this special case, these shouldn't be here, or they should be excluded by the toolchain
            if "clang/staging/include" in file.short_path:
                continue

            data_files.add(file.short_path)

    # Remove C++ toolchain files, they will be provided by the toolchain in the venv.
    py_toolchain_files = ctx.toolchains["@bazel_tools//tools/python:toolchain_type"].py3_runtime.files.to_list()
    toolchain_files = set([x.short_path for x in find_cpp_toolchain(ctx).all_files.to_list() + py_toolchain_files])
    data_files -= toolchain_files
    data_files -= external_python_files

    ctx.actions.write(
        output = output,
        content = json.encode({
            "imports": depset(transitive = transitive_imports).to_list(),
            "files": sorted(external_python_files),
            "data_files": sorted(data_files),
        }),
    )

    return [
        DefaultInfo(files = depset([output])),
    ]

_collect_venv_files = rule(
    implementation = _collect_venv_files_impl,
    attrs = {
        "data": attr.label_list(allow_files = True),
        "deps": attr.label_list(providers = [PyInfo]),
    },
    toolchains = use_cpp_toolchain() + [
        "@bazel_tools//tools/python:toolchain_type",
    ],
)

def modular_py_venv(name, data = [], deps = [], target_compatible_with = []):
    _collect_venv_files(
        name = name + ".collect_venv_files",
        data = data,
        deps = deps,
        target_compatible_with = target_compatible_with,
        testonly = True,
        tags = ["manual"],
    )

    py_binary(
        name = name,
        srcs = ["//bazel/internal:create_venv"],
        main = "//bazel/internal:create_venv.py",
        data = data + [name + ".collect_venv_files"],
        target_compatible_with = target_compatible_with,
        env = {
            "VENV_NAME": ("." + native.package_name() + "/" + name).replace("/", "+"),
            "VENV_MANIFEST": "$(location {})".format(name + ".collect_venv_files"),
        },
        deps = deps,
        tags = ["manual", "no-mypy"],
        testonly = True,
    )
