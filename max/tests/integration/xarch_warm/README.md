# Eager-interpreter GC-sweep warm cache

The eager interpreter compiles a graph-compiler `Model` for each op (matmul,
unary) on first dispatch. On a cold cache that first dispatch lowers and JIT-
compiles a large slice of the kernel library, thousands of kernels, minutes.
This package pays that cost once, at build time, and lets the runtime adopt the
result instead of recompiling, wired so a normal PR's CI can build and check
it.

## How it works

The producer is split into single-purpose build actions so the ~20-min CPU
sweep runs once per host arch, not once per GPU lane. The old unified action
baked the CPU slice into a `--target`-keyed GPU action, so the remote cache
couldn't share it across the b200/h100/a100/mi355 lanes (and it ran again in
the CPU-only action). Now the CPU sweep is one config-independent action every
lane shares.

- `warm_config.py`: the one axis the producers and consumer still pin
  identically: `WARM_DEVICE_COUNT`, the max GPU slots warmed. The host-CPU
  target and GPU arch come from per-platform `select()`s in `BUILD.bazel` (the
  producer records the CPU target in the manifest; the consumer reads it back),
  so the ends agree without a shared target constant.
- `warm_lib.py`: the shared producer loop: for each device slot, build that
  slot's per-device module, compile it on a single-device session, and export
  one MEF (`matmul_cpu.mef`, `matmul_slot_0.mef`, …), plus the stamp/manifest
  writers. Kept import-light so the producers can set the virtual-device knobs
  before the first `_interpreter_ops` import (its device set freezes at import).
- `warm_cpu.py`: sets no virtual accelerator and warms only the CPU slot, into
  a manifest with no `gpu` key and an `accelerators=0` stamp. Takes only
  `--cpu-target`, so its build action is identical across every x86 lane.
- `warm_accelerators.py`: sets the virtual device (`WARM_DEVICE_COUNT` virtual
  accelerators of the `--target` arch, no physical GPU) and warms only the
  accelerator slots, into a manifest fragment (`gpu.arch` + `device_count` + the
  gpu entries) and the GPU-context stamp. It never compiles the CPU slot.
- `compose.py`: merges a `warm_cpu` dir and a `warm_accelerators` dir into one
  GPU-consumer dir: the union of MEFs, one merged `manifest.json` (the CPU
  envelope plus the gpu bits, and both entry lists), and the GPU-context stamp.
  Pure stdlib, it only moves files and merges JSON, no recompile.
- `interp_cache.bzl`: the `warm_interp_cache` rule runs a per-slot producer
  (`warm_cpu`/`warm_accelerators`) as a build action whose declared
  tree-artifact output holds its MEFs + manifest + stamp; `compose_warm_cache`
  runs `compose` over both. Both re-inject the binary's `RunEnvironmentInfo`
  (the `MODULAR_MOJO_MAX_*` kernel-import vars a `py_binary`'s `env` supplies
  only under `bazel run`/`bazel test`, never when exec'd as a `genrule` tool)
  and `cd` into the runfiles tree so those paths resolve, the same recipe as
  `//max/tests/integration/tools:precompile_pipeline`. (A `genrule` build action
  does not work: `genrule.tools` forces an exec transition under which
  `modular_py_binary`'s `$(COMPUTED_IMPORT_PATH)` resolves empty, so the kernel
  import fails before any code runs.)
- `test_interp_warm_cache.py`: the GPU consumer. It takes the `compose`
  directory as a `data` dependency (the build-output-to-test-input edge), fails
  if it isn't populated, checks the warm stamp is adoptable on this real GPU
  box, and dispatches CPU + GPU matmul and unary ops through the eager
  interpreter, asserting each adopts its per-slot MEFs via manifest force-load
  (`session.load_all`, which bypasses compilation and the toolchain cache key)
  with no cold recompile.
- `test_cpu_warm_cache.py`: the CPU consumer. It takes the `warm_cpu`
  directory as a `data` dependency and, on a GPU-less box, force-loads the CPU
  MEFs at the import-time precompile instead of a cold compile.

Each MEF is self-contained, host-ELF kernels and GPU cubins are embedded, so
the consumer force-loads them directly, and the build worker never needs a
physical GPU (virtual-device mode presents `sm_100a` for compilation only).

Run it (Linux only; the eager interpreter tests are macOS-incompatible,
FIXME MOCO-2411):

    bazel test //max/tests/integration/xarch_warm:test_interp_warm_cache

## Caveats

- **x86 GPU builders only.** The sole eager GPU consumer lane is the x86 + B200
  runner, so an aarch64-warmed GPU cache has no validated consumer;
  `warm_accelerators.py` aborts cleanly on a non-x86 host rather than emit an
  unexercised artifact. `warm_cpu.py` has no such guard, aarch64 is a valid
  CPU-only target, and the per-platform CPU-target `select()` carries an
  aarch64 entry (mirroring the mojo toolchain), so adding an aarch64 lane needs
  no edit here.
- **Device arch must be `sm_100a`**, not the bare `sm_100`, `ptxas` rejects
  the bare form for architectures that require the accelerated-features suffix.
- **Force-load is device-count-independent.** The warm emits one single-device
  MEF per slot, so a consumer with `k` GPUs force-loads slots `0..k-1`: a warm
  made for `WARM_DEVICE_COUNT` slots serves any box with `<=` that many GPUs. A
  box needing *more* slots than were warmed falls back to lazy per-target
  compilation (the missing slots were never built).
- **Toolchain adoption is an asserted sentinel**, opt in with
  `MODULAR_EAGER_WARM_ADOPT_ASSERTED=1`, for a closed loop where CI controls
  both ends. The toolchain axis is exact-ABI (no compatibility range), so this
  must stay an explicit opt-in.
- **Host-arch match is coarse; `x86-64-v3` is the floor.** `manifest_adoptable`
  checks `platform.machine()`, which returns `x86_64` for x86-64-v2/v3/v4 alike;
  it does *not* validate the manifest's `cpu_target`. The warm's host-ELF
  kernels are compiled for the `x86-64-v3` target the per-platform CPU
  `select()` pins, so adoption assumes every asserted-opt-in x86 box is at least
  v3: a v2 box would `SIGILL` on a v3 instruction. This holds for the closed CI
  loop (all x86 runners are ≥ v3) and is the operative assumption behind the
  coarse match: the compiled CPU *target* is a floor, not an exact-match key.
- **`set_virtual_cpu_target` grammar.** Only the descriptor form
  (`triple=<triple>;cpu=<name>`) pins the triple; a bare CPU name (`x86-64-v3`)
  only re-tunes `-mcpu` on the host's own triple.
