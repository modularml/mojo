"""Wrapper for mojo_shared_library to add internal logic."""

load("@rules_mojo//mojo:mojo_shared_library.bzl", _upstream_mojo_shared_library = "mojo_shared_library")
load(":cc_transition.bzl", "asan_to_production_config_select", "cc_transition")

# A variant of the upstream mojo_shared_library that, under `--config=asan`,
# runs the production (non-instrumented) compiler -- fast -- while the target
# itself and its dependencies stay in the incoming asan configuration. The
# transition sits on the compiler attribute rather than on the rule; see
# mojo_library.bzl for why, and cc_transition.bzl for the asan-only gating
# (`new_config`).
_transitioned_mojo_shared_library = rule(
    implementation = lambda ctx: ctx.super(),
    parent = _upstream_mojo_shared_library,
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

def mojo_shared_library(
        name,
        use_production_compiler_for_asan = False,
        tags = [],
        **kwargs):
    library_rule = _transitioned_mojo_shared_library if use_production_compiler_for_asan else _upstream_mojo_shared_library
    library_kwargs = {"new_config": asan_to_production_config_select} if use_production_compiler_for_asan else {}
    library_rule(
        name = name,
        tags = ["mojo-fixits"] + tags,
        **(kwargs | library_kwargs)
    )
