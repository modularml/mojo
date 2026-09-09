"""`modular_eager_py_test`: `modular_py_test` + build-warmed eager-GC cache wiring.
"""

load("//bazel:api.bzl", "modular_py_test")

_WARM_PKG = "//max/tests/integration/xarch_warm"

_WARM_DATA = select({
    "//:b200_gpu": [_WARM_PKG + ":compose"],
    "//:h100_gpu": [_WARM_PKG + ":compose"],
    "//:mi355_gpu": [_WARM_PKG + ":compose"],
    "//conditions:default": [_WARM_PKG + ":warm_cpu"],
})

# `XARCH_WARM_RLOCATION` must name the same target `_WARM_DATA` provides, so the
# conftest points `MODULAR_DERIVED_PATH` at the runfiles dir that was staged.
_COMPOSE_RLOC = {"XARCH_WARM_RLOCATION": "$(rlocationpath {}:compose)".format(_WARM_PKG)}
_CPU_RLOC = {"XARCH_WARM_RLOCATION": "$(rlocationpath {}:warm_cpu)".format(_WARM_PKG)}
_WARM_RLOCATION_ENV = select({
    "//:b200_gpu": _COMPOSE_RLOC,
    "//:h100_gpu": _COMPOSE_RLOC,
    "//:mi355_gpu": _COMPOSE_RLOC,
    "//conditions:default": _CPU_RLOC,
})

_WARM_ENV = {
    # Batch-compile the GC matrix once at import, not lazily per dispatch.
    "MAX_EAGER_OP_PRECOMPILE": "1",
    # Opt in to asserted-toolchain force-load adoption of the warm cache above.
    "MODULAR_EAGER_WARM_ADOPT_ASSERTED": "1",
}

_HIDE_ACCELERATORS_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
}

# Warm-adoption pytest plugin.
_PLUGIN_LABEL = _WARM_PKG + ":eager_warm_plugin"
_PLUGIN_MODULE = "eager_warm_plugin"

def modular_eager_py_test(
        name,
        srcs,
        cpu_only = False,
        data = [],
        env = {},
        deps = [],
        ignore_extra_deps = [],
        args = [],
        **kwargs):
    """A `modular_py_test` that adopts the build-time-warmed eager-GC cache.

    Args:
        name: Test target name.
        srcs: Test sources.
        cpu_only: For tests that do not require a GPU. More of a hack to force
          the test to run with `CUDA_VISIBLE_DEVICES=""` and
          `HIP_VISIBLE_DEVICES=""` which allows the test to use cached ops.
        data: Extra runtime deps; merged with the warm cache dir.
        env: Extra env vars; merged over the warm env (caller wins on conflict).
        deps: py deps; merged with the warm-adopt plugin lib.
        ignore_extra_deps: Forwarded; the plugin lib is appended (it is loaded
            via `-p`, never statically imported, so pydeps must not flag it).
        args: Extra pytest args; the `-p` plugin registration is prepended.
        **kwargs: Forwarded verbatim to `modular_py_test` (`gpu_constraints`,
            `size`, `tags`, `per_test_shard_count`, ...).
    """
    if cpu_only:
        warm_data = [_WARM_PKG + ":warm_cpu"]
        warm_env = _CPU_RLOC | _HIDE_ACCELERATORS_ENV
    else:
        warm_data = _WARM_DATA
        warm_env = _WARM_RLOCATION_ENV
    modular_py_test(
        name = name,
        srcs = srcs,
        data = data + warm_data,
        env = _WARM_ENV | warm_env | env,
        deps = deps + [_PLUGIN_LABEL],
        ignore_extra_deps = ignore_extra_deps + [_PLUGIN_LABEL],
        args = ["-p", _PLUGIN_MODULE] + args,
        **kwargs
    )
