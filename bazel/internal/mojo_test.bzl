"""Wrapper around upstream mojo_test to add some internal features."""

load("@rules_mojo//mojo:mojo_test.bzl", _upstream_mojo_test = "mojo_test")
load("//bazel:config.bzl", "ALLOW_UNUSED_TAG")
load("//bazel/internal:config.bzl", "GPU_TEST_ENV", "RUNTIME_SANITIZER_DATA", "get_default_exec_properties", "get_default_test_env", "get_resources_exec_properties", "get_resources_tags", "runtime_sanitizer_env", "validate_gpu_tags")  # buildifier: disable=bzl-visibility
load(":mojo_binary.bzl", "mojo_binary")

def mojo_test(
        name,
        data = [],
        tags = [],
        target_compatible_with = [],
        env = {},
        toolchains = [],
        size = None,
        exec_properties = {},
        **kwargs):
    """A wrapper for mojo_test to handle GPU constraints

    Args:
        name: The test target's name
        data: Runtime dependencies of the test
        tags: Tags to set on the underlying targets
        target_compatible_with: https://bazel.build/extending/platforms#skipping-incompatible-targets
        env: Environment variables to set during the test run
        toolchains: See upstream docs
        size: See upstream test size docs, affects timeout
        exec_properties: See upstream docs
        **kwargs: Everything else passed through to modular_cc_binary with the exception of `size` and `timeout`
    """
    validate_gpu_tags(tags, target_compatible_with)

    default_exec_properties = get_default_exec_properties(tags, target_compatible_with)
    resources_exec_properties = get_resources_exec_properties(name, test = True)
    _upstream_mojo_test(
        name = name,
        tags = ["mojo-fixits"] + tags + get_resources_tags(name),
        data = data + RUNTIME_SANITIZER_DATA,
        target_compatible_with = target_compatible_with,
        toolchains = toolchains + ["//bazel/internal:current_gpu_toolchain"],
        env = GPU_TEST_ENV | runtime_sanitizer_env() | get_default_test_env(exec_properties) | env,
        size = size,
        exec_properties = default_exec_properties | resources_exec_properties | exec_properties,
        **kwargs
    )

    # NOTE: Different from modular_cc_test just to exercise both rules
    mojo_binary(
        name = name + ".debug",
        tags = tags + [
            "manual",
            ALLOW_UNUSED_TAG,
        ],
        data = data,
        target_compatible_with = target_compatible_with,
        toolchains = toolchains + ["//bazel/internal:current_gpu_toolchain"],
        env = env,
        testonly = True,
        **kwargs
    )
