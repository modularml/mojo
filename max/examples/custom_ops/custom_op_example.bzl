"""Custom Op example helpers to reduce boilerplate in BUILD.bazel file."""

load("//bazel:api.bzl", "modular_py_binary", "modular_run_binary_test", "requirement")

def custom_op_example_py_binary(
        name,
        srcs,
        extra_data = [],
        extra_deps = [],
        target_compatible_with = [],
        tags = []):
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
        deps = [
            "//max/python/max/driver",
            "//max/python/max/dtype",
            "//max/python/max/engine",
            "//max/python/max/graph",
            requirement("numpy"),
        ] + extra_deps,
        target_compatible_with = target_compatible_with,
        ignore_extra_deps = [
            requirement("numpy"),  # Provided as a convenience, most examples use it
            # Eager tensor API uses these, but doesn't need the direct dependencies
            "//max/python/max/engine",
            "//max/python/max/graph",
        ],
    )

    # Run each example as a simple non-zero-exit-code test.
    modular_run_binary_test(
        name = name + ".example-test",
        size = "large",
        args = [],
        binary = name,
        tags = ["gpu"] + tags,
        target_compatible_with = target_compatible_with,
    )
