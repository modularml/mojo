"""A helper macro for running python tests with pytest"""

load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//bazel:config.bzl", "ALLOW_UNUSED_TAG")
load("//bazel/internal:config.bzl", "GPU_TEST_ENV", "RUNTIME_SANITIZER_DATA", "env_for_available_tools", "get_default_exec_properties", "get_default_test_env", "get_resources_exec_properties", "get_resources_tags", "runtime_sanitizer_env", "validate_gpu_tags")  # buildifier: disable=bzl-visibility
load("//bazel/pip:pip_requirement.bzl", requirement = "pip_requirement")
load("//bazel/pip/pydeps:pydeps_test.bzl", "pydeps_test")
load(":modular_py_venv.bzl", "modular_py_venv")
load(":mojo_collect_deps_aspect.bzl", "collect_transitive_mojoinfo")
load(":mojo_test_environment.bzl", "mojo_test_environment")
load(":py_imports.bzl", "compute_py_imports")
load(":py_repl.bzl", "py_repl")

def _get_manual_srcs(tags, per_test_tags, srcs):
    # Srcs that default builds skip, so mypy has to see them via a separate
    # library. A no-mypy suppression opts out, at either granularity.
    if "no-mypy" in tags:
        return []

    if "manual" in tags or "postsubmit" in tags:
        return srcs

    result = []
    for src in srcs:
        src_tags = per_test_tags.get(src, [])
        if "no-mypy" in src_tags:
            continue
        if "manual" in src_tags or "postsubmit" in src_tags:
            result.append(src)

    return result

def modular_py_test(
        name,
        srcs,
        deps = [],
        env = {},
        args = [],
        data = [],
        ignore_extra_deps = [],
        ignore_unresolved_imports = [],
        mojo_deps = [],
        tags = [],
        exec_properties = {},
        target_compatible_with = [],
        gpu_constraints = [],
        main = None,
        imports = [],
        per_test_tags = {},
        test_name_prefix = "",
        shard_count = None,
        per_test_shard_count = {},
        **kwargs):
    """Creates a pytest based python test target.

    Args:
        name: The name of the test target
        srcs: The test source files
        deps: py_library deps of the test target
        env: Any environment variables that should be set during the test runtime
        args: Arguments passed to the test execution
        data: Runtime deps of the test target
        ignore_extra_deps: Forwarded to pydeps_test
        ignore_unresolved_imports: Forwarded to pydeps_test
        mojo_deps: mojo_library targets the test depends on at runtime
        tags: Tags added to the py_test target
        exec_properties: https://bazel.build/reference/be/common-definitions#common-attributes
        target_compatible_with: https://bazel.build/extending/platforms#skipping-incompatible-targets
        gpu_constraints: GPU requirements for the tests
        main: If provided, this is the main entry point for the test. If not provided, pytest is used.
        imports: Additional python import paths
        per_test_tags: A mapping of source files to extra tags to apply to that test file.
        test_name_prefix: Prefix added to per-src py_test target names (multi-source only).
        shard_count: Forwarded to the underlying test target.
        per_test_shard_count: Forwarded to the underlying test target for the specified source(s).
        **kwargs: Extra arguments passed through to py_test
    """

    if len(imports) > 1:
        fail("modular_py_test only supports a single import path.")

    if len(set(per_test_tags.keys()) - set(srcs)) != 0:
        fail("keys specified in per_test_tags that are not source files: {}".format(set(per_test_tags.keys()) - set(srcs)))

    if len(set(per_test_shard_count.keys()) - set(srcs)) != 0:
        fail("keys specified in per_test_shard_count that are not source files: {}".format(set(per_test_shard_count.keys()) - set(srcs)))

    if "gpu" in tags and "enable-sanitizers" in tags:
        fail("gpu + sanitizers are able to be run manually, but not in CI. remove `enable-sanitizers`.")

    validate_gpu_tags(tags, target_compatible_with + gpu_constraints)
    toolchains = [
        "//bazel/internal:current_gpu_toolchain",
    ]

    has_test = False
    for src in srcs:
        if name == src.split("/")[0]:
            fail("modular_py_test targets cannot have the same 'name' as a directory: {}. Rename the bazel target or the directory".format(name))
        if src.split("/")[-1].startswith("test_"):
            has_test = True

    if not main and not has_test:
        fail("At least 1 file in modular_py_test must start with 'test_' for pytest to discover them")

    extra_env = runtime_sanitizer_env() | {
        "PYTHONUNBUFFERED": "set",
    }
    extra_data = RUNTIME_SANITIZER_DATA
    transitive_mojo_deps = name + ".mojo_deps"
    collect_transitive_mojoinfo(
        name = transitive_mojo_deps,
        deps_to_scan = deps,
        target_compatible_with = gpu_constraints + target_compatible_with,
        testonly = True,
    )

    env_name = name + ".mojo_test_env"
    toolchains.append(env_name)
    extra_data += [env_name]  # buildifier: disable=list-append
    extra_env |= {
        "MODULAR_MOJO_MAX_COMPILERRT_PATH": "$(COMPILER_RT_PATH)",
        "MODULAR_MOJO_MAX_DRIVER_PATH": "$(MOJO_BINARY_PATH)",
        "MODULAR_MOJO_MAX_IMPORT_PATH": "$(COMPUTED_IMPORT_PATH)",
        "MODULAR_MOJO_MAX_LINKER_DRIVER": "$(MOJO_LINKER_DRIVER)",
        "MODULAR_MOJO_MAX_LLD_PATH": "$(LLD_PATH)",
        "MODULAR_MOJO_MAX_SHARED_LIBS": "$(COMPUTED_LIBS)",
        "MODULAR_MOJO_MAX_SYSTEM_LIBS": "$(MOJO_LINKER_SYSTEM_LIBS)",
    }
    mojo_test_environment(
        name = env_name,
        data = mojo_deps + [transitive_mojo_deps],
        testonly = True,
    )

    default_exec_properties = get_default_exec_properties(tags, gpu_constraints)
    extra_env |= get_default_test_env(exec_properties)

    if "requires-network" in tags:
        # Assume networking is used for huggingface and add the cache
        extra_env |= {"HF_ESCAPES_SANDBOX": "1"}

    test_srcs = [src for src in srcs if src.split("/")[-1].startswith("test_")]
    non_test_srcs = [src for src in srcs if not src.split("/")[-1].startswith("test_")]
    extra_env |= GPU_TEST_ENV

    modular_py_venv(
        name = name + ".venv",
        data = data + extra_data,
        target_compatible_with = gpu_constraints + target_compatible_with,
        deps = deps + [
            requirement("pytest"),
        ],
    )

    py_repl(
        name = name + ".debug",
        data = data + extra_data,
        deps = deps + [
            requirement("pytest"),
            "@rules_python//python/runfiles",
        ],
        direct = False,
        env = env_for_available_tools() | extra_env | env | {
            "DEBUG_SRCS": ":".join(["$(location {})".format(src) for src in srcs]),
            # TODO: This should be PYTHONINSPECT but that doesn't work. We're avoiding args so lldb works without --
            "PYTHONSTARTUP": "$(location //bazel/internal:test_debug_shim.py)",
        },
        srcs = srcs + ["//bazel/internal:test_debug_shim.py"],
        toolchains = toolchains,
        target_compatible_with = gpu_constraints + target_compatible_with,
    )

    if main:
        final_args = args
        final_main = main
    else:
        final_args = [native.package_name(), "-svv", "--color=yes", "--durations=3"] + args
        final_main = "pytest_runner.py"

    manual_srcs = _get_manual_srcs(tags, per_test_tags, srcs)
    if manual_srcs:
        # Non-test srcs are sibling helper modules the manual tests import, so
        # mypy needs them here too (they're already in manual_srcs when the
        # whole target is manual).
        mypy_srcs = manual_srcs + [src for src in non_test_srcs if src not in manual_srcs]

        # TODO: Remove once we run mypy-style lints in a separate test target.
        # Raw py_library, not modular_py_library: the latter loads
        # modular_py_test, so depending back on it would cycle.
        py_library(
            name = name + ".mypy_library",
            data = data + extra_data,
            tags = [ALLOW_UNUSED_TAG, "no-pydeps"],
            deps = deps + [
                requirement("pytest"),
                "@rules_python//python/runfiles",
            ],
            testonly = True,
            srcs = mypy_srcs + ["//bazel/internal:pytest_runner"],
            visibility = ["//visibility:private"],
            imports = compute_py_imports(native.package_name(), imports),
        )

    if len(test_srcs) > 1:
        if shard_count:
            fail("do not use shard_count when there are multiple tests, use per_test_shard_count")

        test_names = []
        for src in test_srcs:
            n_shards = per_test_shard_count.get(src)

            # If a custom main is used, it is responsible for sharding via
            # TEST_SHARD_INDEX and TEST_TOTAL_SHARDS env vars.
            use_shard_plugin = n_shards and not main
            shard_args = ["-p", "pytest-shard"] if use_shard_plugin else []
            test_name = test_name_prefix + src.replace(".py", "")
            test_names.append(test_name)
            py_test(
                name = test_name,
                data = data + extra_data,
                main = final_main,
                args = final_args + shard_args,
                toolchains = toolchains,
                env = env_for_available_tools() | extra_env | env,
                deps = deps + [
                    requirement("pytest"),
                    "@rules_python//python/runfiles",
                ] + (["//bazel/internal:pytest-shard"] if use_shard_plugin else []),
                shard_count = n_shards,
                srcs = [src] + non_test_srcs + ["//bazel/internal:pytest_runner"],
                exec_properties = default_exec_properties | get_resources_exec_properties(test_name, test = True) | exec_properties,
                target_compatible_with = gpu_constraints + target_compatible_with,
                tags = tags + get_resources_tags(test_name) + per_test_tags.get(src, []),
                imports = imports,
                **kwargs
            )

        native.test_suite(
            name = name,
            tests = test_names,
            tags = ["manual"],
        )
    else:
        if per_test_tags:
            fail("Don't use `per_test_tags` if only one source file is specified, use `tags` directly.")
        if per_test_shard_count:
            fail("do not use per_test_shard_count with only one test, use shard_count")

        # If a custom main is used, it is responsible for sharding via
        # TEST_SHARD_INDEX and TEST_TOTAL_SHARDS env vars.
        use_shard_plugin = shard_count and not main
        shard_args = ["-p", "pytest-shard"] if use_shard_plugin else []

        # test_name_prefix intentionally doesn't apply here: single-source
        # collisions happen at the `name` arg, which callers pick themselves.
        py_test(
            name = name,
            data = data + extra_data,
            toolchains = toolchains,
            env = env_for_available_tools() | extra_env | env,
            main = final_main,
            args = final_args + shard_args,
            deps = deps + [
                requirement("pytest"),
                "@rules_python//python/runfiles",
            ] + (["//bazel/internal:pytest-shard"] if use_shard_plugin else []),
            shard_count = shard_count,
            srcs = srcs + ["//bazel/internal:pytest_runner"],
            exec_properties = default_exec_properties | get_resources_exec_properties(name, test = True) | exec_properties,
            target_compatible_with = gpu_constraints + target_compatible_with,
            tags = tags + get_resources_tags(name),
            imports = imports,
            **kwargs
        )

    if "no-pydeps" not in tags:
        pydeps_test(
            name = name + ".pydeps_test",
            data = data,
            srcs = srcs,
            # We provide these as a convenience, okay if not used.
            ignore_extra_deps = ignore_extra_deps + [
                requirement("pytest"),
                "@rules_python//python/runfiles",
            ],
            ignore_unresolved_imports = ignore_unresolved_imports,
            target_compatible_with = select({
                # No point in running these, causes "error replanting symlinks" failures
                "//:asan": ["@platforms//:incompatible"],
                "//:ubsan": ["@platforms//:incompatible"],
                "//conditions:default": [],
            }),
            imports = imports,
            deps = deps + [
                requirement("pytest"),
                "@rules_python//python/runfiles",
            ],
            tags = ["pydeps"],
        )
