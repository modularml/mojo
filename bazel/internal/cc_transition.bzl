"""Transition that runs the Mojo build action with the production compiler under asan.

Applied to the `_mojo_override` compiler attribute of
`_transitioned_mojo_library` / `_transitioned_mojo_shared_library` (gated behind
`use_production_compiler_for_asan`), this flips `//:modular_config` to
`production` ONLY when the incoming config is `asan`. Under `--config=asan` the
Mojo build action then runs the *non-instrumented* compiler -- fast -- while
producing the same artifact (a `.mojoc` / object is config-invariant: neither
`mojo precompile` nor `mojo build` is passed `--sanitize`), instead of the
ASAN-instrumented compiler that makes asan builds slow.

It is an ATTRIBUTE transition, not a rule-level `cfg`, and that distinction is
load-bearing. A rule-level `cfg` reconfigures the target and everything beneath
it, so every cc library under a Mojo library forks into an `-ST-` variant. Those
libraries then reach a depending binary twice -- once through the Mojo graph and
once through an ordinary cc dep -- and rules_cc rejects the same dynamic library
in two configurations outright ("You are trying to link the same dynamic library
... built in a different configuration"). It is also wrong on its face: an asan
binary would link non-instrumented copies of `libMSupportGlobals.so` and friends.
Only the compiler needs reconfiguring, so only the compiler is transitioned; the
target and its deps stay in the incoming asan config and keep their coverage.

The asan-only gating lives in `asan_to_production_config_select` (passed as
`new_config`), which maps `asan` -> `production` and every other config to
ITSELF. So under Bazel's `diff_against_baseline` output-directory naming:

  * Under `--config=asan` the compiler lands on a single shared
    `k8-opt-asan-ST-<hash>` variant, built once with the production compiler and
    shared by every asan consumer.
  * Under any other config the transition output equals the baseline, so Bazel
    does NOT fork an `-ST-` variant: the compiler is the one the toolchain
    already provides.

The transition flips ONLY `//:modular_config` -- moving it off `asan` is what
makes the C++ toolchain drop the ASAN instrumentation. It deliberately leaves the
sanitizer-only link/copt options alone; `@bazel_tools//tools/cpp:link_extra_libs`
in particular cannot be reset here anyway (`link_extra_lib` is a cc_library that
would then link itself, and `config.none()` is a no-op transition object, not an
option-reset sentinel, so it is type-rejected as a dict value).
"""

load("//bazel:config.bzl", "MODULAR_CONFIGS")

def _cc_transition_impl(_settings, attr):
    # `new_config` is resolved by the caller via a select() over the current
    # `//:modular_config` (see `asan_to_production_config_select`): it maps `asan` ->
    # `production` and every other config to ITSELF. So under non-asan builds
    # the output equals the baseline -- a true no-op, so Bazel does NOT fork an
    # `-ST-` variant and those builds are completely unaffected. Reading the
    # config here directly is unreliable (Starlark flags are reset in exec/tool
    # instances by `--incompatible_exclude_starlark_flags_from_exec_config`),
    # which is why the gating lives in a select() resolved against the target
    # config instead.
    if attr.new_config not in MODULAR_CONFIGS:
        fail("expected a value in MODULAR_CONFIGS, got ", attr.new_config)
    return {
        "//:modular_config": attr.new_config,
    }

# Maps the current `//:modular_config` to the config the transition should pin
# to: `asan` -> `production` (run the fast, non-instrumented compiler), every
# other config -> itself (no-op, no `-ST-` fork). Pass as a `mojo_library` /
# `mojo_shared_library`'s `new_config` when `use_production_compiler_for_asan` is set.
asan_to_production_config_select = select({
    "//:modular_config_" + config: ("production" if config == "asan" else config)
    for config in MODULAR_CONFIGS
})

cc_transition = transition(
    implementation = _cc_transition_impl,
    inputs = [],
    outputs = [
        "//:modular_config",
    ],
)
