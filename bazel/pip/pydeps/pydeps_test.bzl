"""Generates a test to enforce that Python dependencies are declared, and to flag unused dependencies."""

load("@rules_python//python:py_info.bzl", "PyInfo")
load("@with_cfg.bzl//with_cfg/private:select.bzl", "decompose_select_elements")  # buildifier: disable=bzl-visibility

def _pydeps_test_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".test_script")

    ctx.actions.symlink(
        output = output,
        target_file = ctx.executable._linter,
        is_executable = True,
    )

    # Construct the args into the test, as MEGA ENVIRONMENT VARIABLES
    # Per dependency:
    # - The label (for reporting) maps to a list of source files it provides, relative to its import path.
    # Our source files (.py, .pyi, .so, .mojo), also relative to our own import path.
    # List of third party deps
    # Ignored extra deps
    # Ignored unresolved imports
    third_party_deps = []
    env = {"DEP_SOURCES": {}}

    for dep in ctx.attr.deps:
        label = str(dep.label)

        # TODO: Remove this exception
        if "rules_python" in dep.label.repo_name:
            third_party_deps.append(label)
            continue

        py_info = dep[PyInfo]
        def_info = dep[DefaultInfo]

        raw_py_sources = [file.short_path for file in py_info.direct_original_sources.to_list()] + [file.short_path for file in py_info.direct_pyi_files.to_list()]

        # TODO: Might need to be more careful here
        raw_non_py_sources = [file.short_path for file in def_info.files.to_list() if file.short_path.endswith(".so") or file.short_path.endswith(".mojo")]
        raw_sources = raw_py_sources + raw_non_py_sources

        if not raw_sources:
            fail("Error: Dependency {} has no source files.".format(dep.label))

        # Whether we like it or not, //foo/bar:baz.py is always importable as foo.bar.baz
        srcs = raw_sources

        # PyInfo doesn't give us the import paths directly, just a list of all of them. Try to
        # infer the correct one by looking for one that is a prefix of one of the source files.
        # NB: We assume there is only one import path.
        import_path = None
        reference_source = raw_sources[0]

        for path in py_info.imports.to_list():
            path = path.removeprefix("_main/")
            if reference_source.startswith(path):
                if import_path:
                    # We have multiple possible import paths, take the most specific one
                    if len(path) > len(import_path):
                        import_path = path
                else:
                    import_path = path
        if import_path:
            srcs += [path.removeprefix(import_path + "/") for path in raw_sources]

        env["DEP_SOURCES"][label] = srcs

    env["IGNORE_EXTRA_DEPS"] = [str(dep.label) for dep in ctx.attr.ignore_extra_deps]
    env["IGNORE_UNRESOLVED_IMPORTS"] = ctx.attr.ignore_unresolved_imports
    env["THIRD_PARTY_DEPS"] = third_party_deps + ctx.attr.pycross_deps
    env["TARGET_LABEL"] = str(ctx.label)

    # Look for compiled extensions, provided by data, and use our own import path.
    # We treat these as source files, except that they don't get their content
    # checked for imports (for hopefully obvious reasons).
    data_sources = []
    for dep in ctx.attr.data:
        for file in dep[DefaultInfo].files.to_list():
            if (
                file.short_path.endswith(".so") and ("cpython" in file.short_path or "mojo" in file.short_path)
            ) or file.short_path.endswith(".mojo"):
                data_sources.append(file.short_path)

    env["TARGET_SOURCES"] = [src.short_path for src in ctx.files.srcs] + data_sources

    if ctx.attr.imports:
        target_import_path = ctx.label.package + "/" + ctx.attr.imports[0]
    else:
        target_import_path = ctx.label.package

    # This one is fine to be absolute
    env["WORKING_DIR"] = target_import_path

    args_file = ctx.actions.declare_file(ctx.label.name + ".args.json")
    ctx.actions.write(args_file, json.encode(env))

    return [
        DefaultInfo(
            executable = output,
            runfiles = ctx.attr._linter[DefaultInfo].default_runfiles.merge(ctx.runfiles(files = ctx.files.srcs + [args_file])),
        ),
        RunEnvironmentInfo(
            environment = {"PYDEPS_TEST_ARGS_FILE": args_file.short_path},
        ),
    ]

def _split_third_party_deps(name, **kwargs):
    deps = kwargs.pop("deps", [])
    pycross_deps = kwargs.pop("pycross_deps", [])
    if pycross_deps:
        fail("This should not be set explicitly, it is computed from deps.")

    new_deps = []
    for in_select, elements in decompose_select_elements(deps):
        if in_select:
            # Rebuild the select() contents, excluding third party deps
            new_select = {}
            for key, values in elements.items():
                new_select_deps = []
                for dep in values:
                    if "pycross" in dep.repo_name:
                        pycross_deps.append(str(dep))
                    else:
                        new_select_deps.append(dep)
                new_select[key] = new_select_deps
            new_deps += select(new_select)
        else:
            for dep in elements:
                if "pycross" in dep.repo_name:
                    pycross_deps.append(str(dep))
                else:
                    new_deps.append(dep)

    return kwargs | {
        "deps": new_deps,
        "pycross_deps": pycross_deps,
    }

pydeps_test = rule(
    implementation = _pydeps_test_impl,
    initializer = _split_third_party_deps,
    attrs = {
        "deps": attr.label_list(
            doc = "List of dependencies.",
            providers = [PyInfo],
        ),
        "pycross_deps": attr.string_list(
            doc = "List of third party dependency labels.",
        ),
        "data": attr.label_list(
            doc = "List of data dependencies.",
            allow_files = True,
        ),
        "ignore_extra_deps": attr.label_list(
            doc = "List of dependency labels to ignore if they are unused.",
        ),
        "ignore_unresolved_imports": attr.string_list(
            doc = "List of import paths to ignore if they are not mapped to a dependency.",
        ),
        "imports": attr.string_list(
            doc = "Import paths for the target source files.",
        ),
        "srcs": attr.label_list(
            doc = "Source files to check imports on. Allows .py, .pyi, .so, and .mojo files, but only checks .py and .pyi files' content.",
            allow_files = True,
        ),
        "_linter": attr.label(
            doc = "The pydeps enforcer script.",
            default = Label("//bazel/pip/pydeps:pydeps_test"),
            cfg = "target",
            executable = True,
        ),
    },
    test = True,
)
