"""Wrapper around upstream mojo_library rule to add documentation generation."""

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@rules_mojo//mojo:mojo_library.bzl", _upstream_mojo_library = "mojo_library")
load("//bazel:config.bzl", "ALLOW_UNUSED_TAG")
load(":cc_transition.bzl", "asan_to_production_config_select", "cc_transition")
load(":mojo_doc.bzl", "mojo_doc")

# A variant of the upstream mojo_library that, under `--config=asan`, runs the
# production (non-instrumented) compiler -- fast -- while the target itself and
# its dependencies stay in the incoming asan configuration.
#
# The transition sits on the compiler attribute rather than on the rule. A
# rule-level `cfg` reconfigures the whole subgraph, which forks every cc library
# under it; a shared library then reaches a depending binary in two
# configurations at once (once through here, once through an ordinary cc dep)
# and the link fails outright. Only the compiler needs reconfiguring, so only
# the compiler is transitioned. See cc_transition.bzl for the asan-only gating
# (`new_config`).
_transitioned_mojo_library = rule(
    implementation = lambda ctx: ctx.super(),
    parent = _upstream_mojo_library,
    attrs = {
        "new_config": attr.string(mandatory = True),
        # Must stay in sync with the toolchain's `mojo`; this is the same
        # compiler, only reconfigured.
        "_mojo_override": attr.label(
            default = Label("//Mojo/tools/mojo"),
            executable = True,
            cfg = cc_transition,
        ),
    },
)

def mojo_library(
        name,
        srcs,
        data = [],
        deps = [],
        validate_missing_docs = True,
        docs_base_path = "",
        docs_title = "",
        docs_hosted_on_mojolang = False,
        show_stability_markers = "none",
        stability_doc_url = "",
        testonly = False,
        visibility = None,
        additional_compiler_inputs = [],
        use_production_compiler_for_asan = False,
        copts = [],
        ignore_deprecated = [],
        import_path = ".",
        tags = []):
    """
    Precompiles sources into a mojoc file.

    Args:
        name: Forwarded to `mojo_library`
        srcs: Forwarded to Forwarded to all subtargets
        deps: Forwarded to Forwarded to all subtargets
        data: Forwarded to `mojo_library`
        validate_missing_docs: Forwarded to `mojo_docs`
        docs_base_path: Forwarded to `mojo_docs`
        docs_title: Forwarded to `mojo_docs`
        docs_hosted_on_mojolang: Forwarded to `mojo_docs`
        show_stability_markers: Forwarded to `mojo_docs`
        stability_doc_url: Forwarded to `mojo_docs`
        testonly: Forwarded to `mojo_library`
        visibility: Forwarded to all subtargets
        additional_compiler_inputs: Forwarded to `mojo_library`
        use_production_compiler_for_asan:
            Use a production build of Mojo to build this target when in asan mode.
            This is a speed optimization if the coverage is not worth the slowdown.
        copts: Forwarded to `mojo_library`
        ignore_deprecated:
            Qualified names (e.g. `Variant.take`) of `@deprecated` declarations
            whose deprecation warning should be suppressed for this target,
            applied to both the library compile and doc generation. All other
            deprecation warnings still surface. `max/kernels` and `Kernels`
            targets that call deprecated Pointer APIs pass
            `IGNORED_POINTER_DEPRECATIONS` (see
            `//max/kernels/src:deprecations.bzl`) here explicitly.
        import_path: Forwarded to `mojo_library`
        tags: Forwarded to all subtargets
    """
    library_rule = _transitioned_mojo_library if use_production_compiler_for_asan else _upstream_mojo_library
    library_kwargs = {"new_config": asan_to_production_config_select} if use_production_compiler_for_asan else {}
    ignore_deprecated_copts = ["--ignore-deprecated=" + decl_name for decl_name in ignore_deprecated]
    library_rule(
        name = name,
        srcs = srcs,
        data = data,
        deps = deps,
        visibility = visibility,
        testonly = testonly,
        tags = ["mojo-fixits"] + tags,
        additional_compiler_inputs = additional_compiler_inputs,
        copts = copts + ignore_deprecated_copts,
        import_path = import_path,
        **library_kwargs
    )

    if not testonly:
        mojo_doc(
            name = name + ".docs",
            srcs = srcs,
            deps = deps,
            validate_missing_docs = validate_missing_docs,
            docs_base_path = docs_base_path,
            docs_title = docs_title,
            docs_hosted_on_mojolang = docs_hosted_on_mojolang,
            show_stability_markers = show_stability_markers,
            stability_doc_url = stability_doc_url,
            copts = ignore_deprecated_copts,
            visibility = visibility,
            tags = [ALLOW_UNUSED_TAG] + tags,
            testonly = testonly,
        )

        build_test(
            name = name + ".docs_test",
            targets = [name + ".docs"],
            tags = ["mojo-docs", "lint-test"] + tags,
            visibility = ["//visibility:private"],
        )
