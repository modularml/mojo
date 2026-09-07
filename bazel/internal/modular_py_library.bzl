"""Wrapper macro for py_library"""

load("@rules_python//python:defs.bzl", "py_library")
load("//bazel/pip/pydeps:pydeps_test.bzl", "pydeps_test")
load(":modular_py_test.bzl", "modular_py_test")
load(":py_imports.bzl", "compute_py_imports")

def modular_py_library(
        name,
        visibility = None,
        ignore_extra_deps = [],
        ignore_unresolved_imports = [],
        imports = [],
        test_docstring_examples = False,
        docstring_example_deps = [],
        docstring_example_shard_count = None,
        tags = [],
        **kwargs):
    """Creates a py_library target

    Args:
        name: The name of the underlying py_library
        visibility: The visibility of the target, defaults to public
        ignore_extra_deps: Forwarded to pydeps_test
        ignore_unresolved_imports: Forwarded to pydeps_test
        imports: The imports path. For max/python/max packages, this is
            automatically computed and should not be passed.
        test_docstring_examples: If True, generate a companion
            <name>.docstring_examples pytest target that runs Sybil on every
            docstring code-block example in this library's sources.
        docstring_example_deps: Extra runtime deps the examples import but the
            library itself cannot depend on (e.g. engine, which depends on
            graph). Ignored unless test_docstring_examples is True.
        docstring_example_shard_count: Shards for the docstring-example test.
            Each example compiles its own graph and CI clamps every test to
            800s, so set this for packages with many examples. Sharding is
            per example (a visible code block plus its invisible checks), so
            every example in a sharded package must be self-contained.
        tags: Tags to add to the target
        **kwargs: Extra arguments passed through to py_library
    """
    imports = compute_py_imports(native.package_name(), imports)

    if "manual" in tags:
        fail("modular_py_library targets cannot be manual. Remove 'manual' from the tags list.")

    py_library(
        name = name,
        visibility = visibility,
        imports = imports,
        tags = tags,
        **kwargs
    )

    if "no-pydeps" not in tags:
        pydeps_test(
            name = name + ".pydeps_test",
            deps = kwargs.get("deps", []),
            data = kwargs.get("data", []),
            ignore_extra_deps = ignore_extra_deps,
            ignore_unresolved_imports = ignore_unresolved_imports,
            target_compatible_with = select({
                # No point in running these, causes "error replanting symlinks" failures
                "//:asan": ["@platforms//:incompatible"],
                "//:ubsan": ["@platforms//:incompatible"],
                "//conditions:default": [],
            }),
            imports = imports if imports != None else [],
            srcs = kwargs.get("srcs", []) + kwargs.get("pyi_srcs", []),
            tags = ["pydeps"],
        )

    if test_docstring_examples:
        # Scope collection to this target's own package directory rather than
        # the whole max/python/max tree, so the test only runs examples from
        # this package (its dependencies live in other package directories and
        # are not collected). Pass the directory, not individual source files:
        # an explicit file argument is an "initpath" that pytest's builtin
        # Python collector imports as a test module (bypassing python_files),
        # which would fail on any optional top-level dependency. Directory
        # recursion instead lets sybil_collect's pytest_ignore_collect skip
        # private and excluded files before they are imported.
        # Sharding is implemented in sybil_collect, per example group;
        # pytest_runner forwards bazel's shard env vars to it.
        # docstring_example_deps put their sources in this test's runfiles.
        # A dep nested under this package's directory would be collected a
        # second time by the directory recursion above (its examples already
        # run in its own package's test), so exclude it from collection
        # while retaining it in runfiles for imports.
        nested_dep_packages = [
            Label(dep).package
            for dep in docstring_example_deps
            if Label(dep).package.startswith(native.package_name() + "/")
        ]
        modular_py_test(
            name = name + ".docstring_examples",
            timeout = "long",
            srcs = [],
            main = "pytest_runner.py",
            shard_count = docstring_example_shard_count,
            # CI runs without network access; fail network-dependent
            # examples on the author's machine instead of in presubmit.
            env = {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            args = [
                native.package_name(),
                "-p",
                "sybil_collect",
                "--import-mode=importlib",
                "-o",
                "consider_namespace_packages=true",
                # Surface the slowest examples so cost creep stays visible.
                "--durations=20",
            ] + ["--ignore=" + pkg for pkg in nested_dep_packages],
            deps = [
                ":" + name,
                "//max/tests/docstring_examples:sybil_collect",
            ] + docstring_example_deps,
            tags = ["no-pydeps"],
        )
