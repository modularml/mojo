"""A helper macro for python scripts which helps setup various runtime dependencies."""

load("@rules_python//python:defs.bzl", "py_binary")
load("//bazel:config.bzl", "ALLOW_UNUSED_TAG")
load("//bazel/internal:config.bzl", "RUNTIME_SANITIZER_DATA", "env_for_available_tools", "runtime_sanitizer_env")  # buildifier: disable=bzl-visibility
load("//bazel/pip/pydeps:pydeps_test.bzl", "pydeps_test")
load(":modular_py_library.bzl", "modular_py_library")
load(":modular_py_venv.bzl", "modular_py_venv")
load(":mojo_collect_deps_aspect.bzl", "collect_transitive_mojoinfo")
load(":mojo_test_environment.bzl", "mojo_test_environment")
load(":py_repl.bzl", "py_repl")

def modular_py_binary(
        name,
        srcs,
        main = None,
        env = {},
        data = [],
        deps = [],
        ignore_extra_deps = [],
        ignore_unresolved_imports = [],
        mojo_deps = [],
        toolchains = [],
        imports = [],
        tags = [],
        args = [],
        testonly = False,
        target_compatible_with = [],
        **kwargs):
    """Creates a pytest based python test target.

    Args:
        name: The name of the test target
        srcs: The test source files
        main: See upstream py_binary docs
        env: Any environment variables that should be set during the test runtime
        data: Runtime deps of the test target
        deps: Python deps of the target
        mojo_deps: mojo_library targets the test depends on at runtime
        toolchains: See upstream py_binary docs
        ignore_extra_deps: Forwarded to pydeps_test
        ignore_unresolved_imports: Forwarded to pydeps_test
        imports: See upstream py_binary docs
        tags: See upstream py_binary docs
        args: See upstream py_binary docs
        testonly: Only test targets can depend on this target
        target_compatible_with: https://bazel.build/extending/platforms#skipping-incompatible-targets
        **kwargs: Extra arguments passed through to py_binary
    """
    if len(imports) > 1:
        fail("modular_py_binary only supports a single import path.")

    mojo_test_env_name = name + ".mojo_test_env"
    extra_toolchains = [
        "@//bazel/internal:current_gpu_toolchain",
        mojo_test_env_name,
    ]
    extra_data = RUNTIME_SANITIZER_DATA + [
        mojo_test_env_name,
    ]
    extra_env = {
        "PYTHONUNBUFFERED": "set",
        "MODULAR_MOJO_MAX_COMPILERRT_PATH": "$(COMPILER_RT_PATH)",
        "MODULAR_MOJO_MAX_DRIVER_PATH": "$(MOJO_BINARY_PATH)",
        "MODULAR_MOJO_MAX_IMPORT_PATH": "$(COMPUTED_IMPORT_PATH)",
        "MODULAR_MOJO_MAX_LINKER_DRIVER": "$(MOJO_LINKER_DRIVER)",
        "MODULAR_MOJO_MAX_LLD_PATH": "$(LLD_PATH)",
        "MODULAR_MOJO_MAX_SHARED_LIBS": "$(COMPUTED_LIBS)",
        "MODULAR_MOJO_MAX_SYSTEM_LIBS": "$(MOJO_LINKER_SYSTEM_LIBS)",
    } | runtime_sanitizer_env()

    transitive_mojo_deps = name + ".mojo_deps"
    collect_transitive_mojoinfo(
        name = transitive_mojo_deps,
        deps_to_scan = deps,
        target_compatible_with = target_compatible_with,
        testonly = testonly,
    )

    mojo_test_environment(
        name = mojo_test_env_name,
        data = mojo_deps + [transitive_mojo_deps],
        short_path = True,
        target_compatible_with = target_compatible_with,
        testonly = testonly,
    )

    py_binary(
        name = name,
        data = extra_data + data,
        deps = deps + [
            "@//bazel/internal:bazel_sitecustomize",  # py_repl adds this automatically
        ],
        srcs = srcs,
        main = main,
        env = env_for_available_tools() | extra_env | env | {"MODULAR_CANNOT_DEBUG": "1"},
        toolchains = extra_toolchains + toolchains,
        imports = imports,
        tags = tags,
        args = args,
        testonly = testonly,
        target_compatible_with = target_compatible_with,
        **kwargs
    )

    # This library exists only so mypy sees srcs that the manual binary hides
    # from default builds, so it must honor the caller's mypy suppression.
    if "manual" in tags and "no-mypy" not in tags:
        # TODO: Remove once we run mypy-style lints in a separate test target
        modular_py_library(
            name = name + ".mypy_library",
            data = data + extra_data,
            toolchains = toolchains,
            deps = deps + [
                "@//bazel/internal:bazel_sitecustomize",
            ],
            testonly = True,
            # Pydeps test is added below
            tags = [ALLOW_UNUSED_TAG, "no-pydeps"],
            srcs = srcs,
            visibility = ["//visibility:private"],
            imports = imports,
            # NOTE: Intentionally exclude other attrs that shouldn't matter for mypy
        )

    modular_py_venv(
        name = name + ".venv",
        data = extra_data + data,
        deps = deps + [
            "@//bazel/internal:bazel_sitecustomize",  # py_repl adds this automatically
        ],
    )

    py_repl(
        name = name + ".debug",
        data = extra_data + data,
        deps = deps,
        direct = False,
        env = env_for_available_tools(location_specifier = "execpath") | extra_env | env,
        args = [native.package_name() + "/" + (main or srcs[0])],
        srcs = srcs,
        toolchains = extra_toolchains + toolchains,
        **kwargs
    )

    if "no-pydeps" not in tags:
        pydeps_test(
            name = name + ".pydeps_test",
            srcs = srcs,
            data = data,
            ignore_extra_deps = ignore_extra_deps,
            ignore_unresolved_imports = ignore_unresolved_imports,
            target_compatible_with = select({
                # No point in running these, causes "error replanting symlinks" failures
                "//:asan": ["@platforms//:incompatible"],
                "//:ubsan": ["@platforms//:incompatible"],
                "//conditions:default": [],
            }),
            imports = imports,
            deps = deps,
            tags = ["pydeps"],
        )
