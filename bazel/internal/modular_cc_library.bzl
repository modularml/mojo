"""Wrapper for the upstream cc_library rule to apply modular specific defaults."""

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(":modular_clang_tidy_test.bzl", "modular_clang_tidy_test")

def modular_cc_library(
        name,
        srcs = [],
        deps = [],
        internal_deps = [],
        hdrs = [],
        copts = [],
        data = [],
        additional_compiler_inputs = [],
        includes = [],
        include_prefix = None,
        strip_include_prefix = None,
        visibility = None,
        tags = [],
        **kwargs):
    """A wrapper for cc_library

    https://bazel.build/reference/be/c-cpp#cc_library

    Args:
        name: The name of the underlying cc_library
        srcs: See cc_library docs
        deps: Deps required to compile the target
        internal_deps: Same as `deps`, but excluded for external builds.
        hdrs: See cc_library docs
        copts: See cc_library docs
        data: Files that should be available at runtime
        additional_compiler_inputs: Non-source files that must be available when compiling the library
        includes: See cc_library docs
        include_prefix: See cc_library docs
        strip_include_prefix: See cc_library docs
        visibility: See cc_library docs
        tags: See cc_library docs
        **kwargs: Everything else passed through to cc_library without processing
    """
    default_strip_include_prefix = "include"
    has_either_prefix = bool(strip_include_prefix or include_prefix)
    if has_either_prefix:
        default_strip_include_prefix = None

    if not hdrs or not hdrs[0].startswith("include/"):
        default_strip_include_prefix = None

    extra_includes = []
    if default_strip_include_prefix:
        extra_includes.append(default_strip_include_prefix)
    cc_library(
        name = name,
        hdrs = hdrs,
        deps = deps + internal_deps,
        srcs = srcs,
        data = data,
        copts = copts,
        strip_include_prefix = strip_include_prefix,
        additional_compiler_inputs = additional_compiler_inputs,
        includes = includes + extra_includes,
        include_prefix = include_prefix,
        visibility = visibility,
        tags = tags,
        **kwargs
    )

    modular_clang_tidy_test(
        name = name,
        hdrs = hdrs,
        srcs = srcs,
        copts = copts,
        tags = tags,
        additional_compiler_inputs = additional_compiler_inputs,
    )
