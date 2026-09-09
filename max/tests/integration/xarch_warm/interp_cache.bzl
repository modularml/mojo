"""Build actions that warm the eager-interpreter GC sweep into a cache dir.

Two rules:

- ``warm_interp_cache`` runs a per-slot producer binary (``warm_cpu`` or
  ``warm_accelerators``) as a build action with ``MODULAR_DERIVED_PATH`` pointed
  at a declared tree-artifact output; the sweep exports one MEF per warmed
  device slot + a manifest + a warm stamp.
- ``compose_warm_cache`` merges a ``warm_cpu`` output dir and a
  ``warm_accelerators`` output dir into one GPU-consumer cache dir (equivalent
  in shape to the old unified warm), so the CPU sweep is a single lane-shared
  action rather than baked into each lane-keyed GPU action.

A consumer test depends on the produced directory via ``data`` and points the
same ``MODULAR_DERIVED_PATH`` at it, the real build-action-output ->
test-input dependency edge, giving CI a warmed cache produced once at build
time.

Both rules reuse the ``//max/tests/integration/tools:precompile_pipeline``
recipe: a ``py_binary``'s ``env`` block (which carries the ``MODULAR_MOJO_MAX_*``
kernel-import vars) is only applied under ``bazel run``/``bazel test``, never
when the binary is exec'd as a ``genrule`` tool, so read the binary's
``RunEnvironmentInfo`` and re-inject it, then ``cd`` into the runfiles tree so
the short_path values in those vars resolve.
"""

load("@cfg_workaround.bzl", "CFG_WORKAROUND")

def _warm_interp_cache_impl(ctx):
    derived_dir = ctx.actions.declare_directory(ctx.attr.name + "_derived")

    binary = ctx.attr.binary[DefaultInfo].files_to_run
    env = dict(ctx.attr.binary[RunEnvironmentInfo].environment)

    # No device hiding: warm_accelerators uses MAX's virtual-device knobs to
    # compile the GPU sweep with no physical GPU, so the worker's real
    # accelerators are irrelevant (warm_cpu sets none).

    args = ctx.actions.args()
    args.add(binary.executable)
    args.add(derived_dir.path)

    # Optional --target: an empty target is a CPU-only warm (warm_cpu), the
    # producer then sets no virtual device, warms only the CPU slot, and writes
    # a manifest with no "gpu" key.
    if ctx.attr.target:
        args.add("--target")
        args.add(ctx.attr.target)
    args.add("--cpu-target")
    args.add(ctx.attr.cpu_target)

    # Optional --family: warm only this GC family; empty warms all.
    if ctx.attr.family:
        args.add("--family")
        args.add(ctx.attr.family)

    # The binary's env vars (e.g. MODULAR_MOJO_MAX_IMPORT_PATH) hold short_path
    # values that resolve relative to the runfiles root, but a build action's
    # CWD is the execroot, so make MODULAR_DERIVED_PATH absolute up front,
    # then cd into the runfiles dir before running.
    ctx.actions.run_shell(
        command = """\
set -e
EXE="$PWD/$1"; shift
export MODULAR_DERIVED_PATH="$PWD/$1"; shift
mkdir -p "$MODULAR_DERIVED_PATH"
cd "${EXE}.runfiles/_main"
"$EXE" "$@"
""",
        arguments = [args],
        tools = [binary],
        use_default_shell_env = True,
        env = env,
        outputs = [derived_dir],
        mnemonic = "WarmInterpCache",
        progress_message = "Warming eager interpreter GC cache %{output}",
    )

    return [DefaultInfo(files = depset([derived_dir]))]

warm_interp_cache = rule(
    doc = "Runs the eager GC sweep as a build action; outputs a warmed cache dir.",
    implementation = _warm_interp_cache_impl,
    attrs = {
        "binary": attr.label(
            mandatory = True,
            executable = True,
            # Deferred: the sweep uses only the graph compiler, never the op .so
            # kernels, but importing the sweep-builders runs
            # _interpreter_ops/__init__, which eagerly imports every op module,
            # so the producer builds op .so's it never uses. Splitting gc_sweeps
            # + gc_compile out of that init would drop the dep.
            cfg = CFG_WORKAROUND,
            doc = "The per-slot producer modular_py_binary to run " +
                  "(warm_cpu or warm_accelerators).",
        ),
        "target": attr.string(
            doc = "Virtual GPU target 'api:arch' (e.g. 'cuda:sm_100a'), " +
                  "passed to the producer as --target. Empty (the default) warms " +
                  "CPU-only: no --target is passed and only the CPU slot is warmed.",
        ),
        "cpu_target": attr.string(
            mandatory = True,
            doc = "Virtual host-CPU target descriptor (e.g. " +
                  "'triple=x86_64-unknown-linux-gnu;cpu=x86-64-v3'), passed to " +
                  "the producer as --cpu-target. Sourced from a select() " +
                  "mirroring the mojo toolchain's per-platform target.",
        ),
        "family": attr.string(
            doc = "Warm only this GC family, passed as --family (empty warms " +
                  "all). The BUILD loops _WARM_FAMILIES to make one fragment " +
                  "per family; the producer validates the name against " +
                  "GC_FAMILIES.",
        ),
    },
)

def _compose_warm_cache_impl(ctx):
    out_dir = ctx.actions.declare_directory(ctx.attr.name + "_derived")

    binary = ctx.attr.binary[DefaultInfo].files_to_run
    env = dict(ctx.attr.binary[RunEnvironmentInfo].environment)

    # Each warm fragment (a per-family CPU or accelerator warm) is a single
    # tree-artifact dir; the compose binary unions them all.
    warm_dirs = [w[DefaultInfo].files.to_list()[0] for w in ctx.attr.warms]

    args = ctx.actions.args()
    args.add(binary.executable)
    args.add(out_dir.path)
    for warm_dir in warm_dirs:
        args.add("--warm-path")
        args.add(warm_dir.path)

    # Absolutize each fragment dir (execroot-relative) before cd'ing into the
    # runfiles, then rebuild them as repeated --warm. POSIX only (no bash arrays).
    ctx.actions.run_shell(
        command = """\
set -e
EXE="$PWD/$1"; shift
export MODULAR_DERIVED_PATH="$PWD/$1"; shift
mkdir -p "$MODULAR_DERIVED_PATH"
ROOT="$PWD"
set -- "$@" __END__
while [ "$1" != "__END__" ]; do
  shift
  dir="$1"; shift
  set -- "$@" --warm "$ROOT/$dir"
done
shift
cd "${EXE}.runfiles/_main"
"$EXE" "$@"
""",
        arguments = [args],
        tools = [binary],
        use_default_shell_env = True,
        env = env,
        inputs = warm_dirs,
        outputs = [out_dir],
        mnemonic = "ComposeInterpWarmCache",
        progress_message = "Composing eager interpreter GC cache %{output}",
    )

    return [DefaultInfo(files = depset([out_dir]))]

compose_warm_cache = rule(
    doc = "Merges N per-family warm fragments (CPU and/or accelerator) into " +
          "one consumer cache dir (union of MEFs + merged manifest + the " +
          "highest-accelerator warm stamp).",
    implementation = _compose_warm_cache_impl,
    attrs = {
        "binary": attr.label(
            mandatory = True,
            executable = True,
            cfg = CFG_WORKAROUND,
            doc = "The compose modular_py_binary to run.",
        ),
        "warms": attr.label_list(
            mandatory = True,
            doc = "Warm fragment dirs (per-family CPU and/or accelerator) to " +
                  "union into one consumer cache dir: CPU-only for the CPU " +
                  "consumer, CPU + accelerator for the GPU consumer.",
        ),
    },
)
