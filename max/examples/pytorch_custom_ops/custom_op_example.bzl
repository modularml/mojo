"""Custom Op example helpers to reduce boilerplate in BUILD.bazel file."""

load("//bazel:api.bzl", "modular_py_binary", "modular_run_binary_test")

def custom_op_example_py_binary(
        name,
        srcs,
        create_test = True,
        extra_data = [],
        **kwargs):
    modular_py_binary(
        name = name,
        srcs = srcs,
        data = [
            ":kernel_sources",
        ] + extra_data,
        imports = ["."],
        mojo_deps = [
            "//max:layout",
            "//max:extensibility",
            "@mojo//:std",
        ],
        visibility = ["//visibility:private"],
        testonly = True,
        **kwargs
    )

    # Run each example as a simple non-zero-exit-code test.
    # The test inherits target_compatible_with from the binary dependency.
    if create_test:
        modular_run_binary_test(
            name = name + ".example-test",
            args = [],
            binary = name,
        )
