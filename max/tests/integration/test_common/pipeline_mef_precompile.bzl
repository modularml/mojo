"""A build action that precompiles a whole pipeline's graphs to reusable MEFs.

The sibling of ``mef_precompile.bzl``: that one suits a test that builds its own
``max.graph.Graph`` and can hand a builder to a producer, this one suits a test
that drives a *pipeline* (a `max generate` / `max serve` CLI test, say) and so
never sees the graphs. A CPU build action runs the pipeline under MAX's
virtual-device knobs with ``--export-mefs`` pointed at a declared output
directory, which makes every graph the pipeline compiles land there as a MEF plus
a manifest. The GPU consumer passes the same directory to ``--precompiled-mefs``
and initializes those artifacts instead of compiling.

Note this is *not* the MEF cache. Reusing artifacts by path avoids matching a
compile key that covers the host CPU target, kernel-package contents and the
build configuration -- none of which agree between a build action and a test --
and a mismatch raises rather than silently falling back to a full recompile. The
consumer still has to build the same graphs, so config the pipeline derives from
device memory must be pinned on both sides; the manifest check reports it when it
is not.

Public API:

- ``precompiled_pipeline_mefs(name, producer, ...)``: the macro a test calls.
  The arch to compile for comes from the mojo toolchain, so the call site names
  neither it nor the host-CPU target.

The rule reuses the ``precompile_pipeline.bzl`` recipe: a ``py_binary``'s ``env``
block carries the ``MODULAR_MOJO_MAX_*`` kernel-import vars only under ``bazel
run``/``test``, never when exec'd as a build tool, so read the binary's
``RunEnvironmentInfo`` and re-inject it, then ``cd`` into the runfiles tree so
those short_path vars resolve.
"""

def _targets_from_mojo_toolchain(ctx):
    """Reads the GPU and host-CPU targets to compile for off the mojo toolchain.

    Same derivation as ``mef_precompile.bzl``, so a call site names neither: the
    toolchain already describes the lane being built for, and the artifacts have
    to agree with it.

    Args:
        ctx: The rule context.

    Returns:
        The GPU target as ``"api:arch"`` and the host-CPU codegen descriptor.
    """
    toolchain = ctx.toolchains["@rules_mojo//:toolchain_type"].mojo_toolchain_info

    target = None
    cpu_target = None
    for copt in toolchain.copts:
        if copt.startswith("--target-accelerator="):
            target = copt.removeprefix("--target-accelerator=")
        elif copt.startswith("--target-cpu="):
            cpu_target = copt.removeprefix("--target-cpu=")

    # A pipeline compiles for an accelerator or not at all, so say so here rather
    # than hand the producer a target of `None` and let it fail confusingly. A
    # target gated to a GPU lane never reaches this.
    if not target:
        fail(
            "the mojo toolchain names no target accelerator, so there is " +
            "nothing to precompile for; gate this target to a GPU lane",
        )

    # Mojo and the graph compiler spell the vendor differently.
    target = target.replace("nvidia", "cuda")
    target = target.replace("amdgpu", "hip")
    return target, cpu_target

def _precompiled_pipeline_mefs_impl(ctx):
    mef_dir = ctx.actions.declare_directory(ctx.attr.name + "_mefs")

    target, cpu_target = _targets_from_mojo_toolchain(ctx)

    binary = ctx.attr.producer[DefaultInfo].files_to_run
    env = dict(ctx.attr.producer[RunEnvironmentInfo].environment)

    # The pipeline is named by HuggingFace repo id, so the producer has to
    # resolve the model's config and weights -- but a build action must not
    # reach the network to do it, and the build platforms rightly run with
    # `dockerNetwork = off`. The executors mount the shared HuggingFace cache
    # read-only at the default location, and a flag set names an explicit
    # revision, so offline mode resolves straight to that snapshot. Same bargain
    # as the sibling rule in `precompile_pipeline.bzl`; a machine whose cache is
    # missing the pinned revision has to populate it before building this.
    env["HF_HUB_OFFLINE"] = "1"

    args = ctx.actions.args()
    args.add(binary.executable)
    args.add(mef_dir.path)
    args.add("--target")
    args.add(target)
    args.add("--cpu-target")
    args.add(cpu_target)
    args.add_all(ctx.attr.producer_args)

    # The binary's env vars hold short_path values that resolve relative to the
    # runfiles root, but a build action's CWD is the execroot: absolutize the
    # output path up front, point MODULAR_DERIVED_PATH at a throwaway scratch dir
    # (MAX's compile caches land there rather than in a declared output), then cd
    # into the runfiles dir before running.
    ctx.actions.run_shell(
        command = """\
set -e
EXE="$PWD/$1"; shift
OUT="$PWD/$1"; shift
export MODULAR_DERIVED_PATH="$PWD/pipeline_mef_scratch"
mkdir -p "$MODULAR_DERIVED_PATH"
cd "${EXE}.runfiles/_main"
"$EXE" --out "$OUT" "$@"
""",
        arguments = [args],
        tools = [binary],
        use_default_shell_env = True,
        env = env,
        outputs = [mef_dir],
        mnemonic = "PrecompilePipelineMefs",
        progress_message = "Precompiling pipeline MEFs %{output}",
    )

    return [DefaultInfo(files = depset([mef_dir]))]

_precompiled_pipeline_mefs = rule(
    doc = "Compiles a pipeline's graphs to MEFs as a CPU build action.",
    implementation = _precompiled_pipeline_mefs_impl,
    attrs = {
        "producer": attr.label(
            mandatory = True,
            executable = True,
            # Target config rather than the usual CFG_WORKAROUND exec config, so
            # the artifacts are built by the same toolchain and kernel packages
            # the consumer runs with. Consumer platforms here are linux-x86_64,
            # which the build workers are too, so the binary still runs.
            cfg = "target",
            doc = "A modular_py_binary that runs a pipeline with the " +
                  "consumer's flags. Takes --out, --target and --cpu-target.",
        ),
        "producer_args": attr.string_list(
            doc = "Extra arguments for the producer, naming which pipeline " +
                  "configuration to compile for.",
        ),
    },
    toolchains = [
        "@rules_mojo//:toolchain_type",
    ],
)

def precompiled_pipeline_mefs(
        name,
        producer,
        producer_args = [],
        testonly = True,
        **kwargs):
    """Precompiles a pipeline's graphs to MEFs on a CPU build action.

    The consumer test depends on ``:name`` via ``data``, reads
    ``$(rootpath :name)`` out of its environment, and passes it to the pipeline
    as ``--precompiled-mefs``.

    Args:
        name: Target name; produces a ``<name>_mefs`` directory of MEFs plus a
            ``manifest.json``.
        producer: A modular_py_binary that runs the pipeline (label).
        producer_args: Extra arguments for the producer, naming which pipeline
            configuration to compile for.
        testonly: Whether the target is test-only. Defaults to ``True``.
        **kwargs: Common attrs (visibility, tags, target_compatible_with, ...).
    """
    _precompiled_pipeline_mefs(
        name = name,
        producer = producer,
        producer_args = producer_args,
        testonly = testonly,
        **kwargs
    )
