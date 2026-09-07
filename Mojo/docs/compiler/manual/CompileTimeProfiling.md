# Compile time profiling

When a Mojo compile is slow, the first question is which part of the compiler
is spending the time. This doc covers the flags that answer that, how to get a
report for a real model, and how to turn the report into a breakdown you can
act on.

## The flags

| Flag                         | What it does                                                                                   |
|------------------------------|------------------------------------------------------------------------------------------------|
| `--mlir-timing`              | Times every MLIR pass and analysis.                                                            |
| `--llvm-timing`              | Times every LLVM pass and analysis.                                                            |
| `--mlir-timing-display MODE` | `tree` (default) nests by pipeline structure; `list` aggregates per pass name, sorted by time. |

See them and their full help text with `mojo build --help-hidden` or
`mojo run --help-hidden`. Their definitions live in
`KGEN/tools/mojo/Common/CompilationOptions.td`.

Four things to know before reading any report they produce:

1. Both reports go to **stderr** when compilation finishes, so capture with
   `2>`. For `mojo run` they print before the program starts.
2. `--llvm-timing` forces single threaded compilation. LLVM's timers are
   process global and not thread safe, so the flag sets `numThreads = 1` and
   overrides `--num-threads`. The report therefore measures total CPU work, not
   the wall time of a parallel build.
3. A warm cache produces an empty or partial report. Passes served from the
   compilation cache never run, so a warm cache reports the parse and little
   else, and LLVM reports nothing at all when the object code is cached. Point
   `MODULAR_CACHE_DIR` at an empty directory to time a full pipeline.
4. `list` display mode merges all pipelines, so it cannot show one offload
   target on its own. Prefer the default `tree`.

## What the report contains

Three sections, in this order:

1. `MLIR pass timing (--mlir-timing)` — a tree of MLIR passes.
2. `LLVM pass timing (--llvm-timing): offload <triple>` — one per accelerator
   target.
3. `LLVM pass timing (--llvm-timing): host <triple>`.

Compiling for an accelerator runs the compiler again on the kernels, so the
accelerator work shows up in **both** halves of the report, in two different
shapes.

Its MLIR passes are inside the MLIR section, as a scope row that opens a
subtree under `ElaborateGenerators`:

```text
  119.7262 ( 45.1%)  ElaborateGenerators
  101.8517 ( 38.4%)  ===--- offload nvptx64-nvidia-cuda sm_100a ... (also in host) ===
   21.9535 (  8.3%)    ElaborateGenerators
   18.3007 (  6.9%)    AutomaticInline
    ...
    0.3545 (  0.1%)  SetFastMathFlags
    ...
  -59.0020 (-22.2%)  Rest
  265.5347 (100.0%)  Total
```

The rows indented one level under the scope are the accelerator's own MLIR
passes — a second `ElaborateGenerators`, a second `AutomaticInline`, and so on,
because it is the same pipeline running again on the kernels. The scope row sits
at the same indent as `ElaborateGenerators`, but the `(also in host)` tag says
its time is counted inside that pass, not next to it. The row after the subtree,
`SetFastMathFlags` here, is back at the outer level and is host work again.

MLIR closes the report with two rows of its own. `Total` is the timing root, so
it is the compile time. `Rest` is whatever time inside that root no pass
claimed, which MLIR computes as the root minus the sum of its children — so it
goes negative, as it does here, whenever those children double count something.
Both rows are covered below:
[`Rest` is host code generation](#rest-is-host-code-generation-not-noise) for
what the time actually is, and
[each accelerator scope is counted twice](#each-accelerator-scope-is-counted-twice)
for why the printed value is `-59.0020` rather than `42.85`.

Its LLVM passes are not in that subtree. They land in the separate
`offload <triple>` LLVM section, even though the time was spent inside
`ElaborateGenerators` too. So to total the accelerator pipeline, add the MLIR
subtree and that LLVM section.

The three sections are not siblings and their totals must not be added: the
MLIR root spans the whole compile including code generation. See
[Reading the numbers by hand](#reading-the-numbers-by-hand) for the
consequences.

## Timing a model

Two steps: get the Mojo that the graph compiler emits for the model, then
compile that with the timing flags. The example uses gemma-4-31B.

### 1. Get the emitted Mojo

```bash
MODULAR_DEBUG=ir-output-dir=/tmp/gemma4/out-dir \
./bazelw run //max/python/max/_entrypoints:pipelines -- warm-cache \
  --model-path google/gemma-4-31B-it --target cuda:sm_100a
```

This runs the whole pipeline, so it fails without the matching GPU. Run it
anyway: the emitted files land before the failure. The output directory holds
the graph IR at each stage plus two Mojo files:

- `gemma4_vision+gemma4_language.mojo` — about 3 MB, the file to compile.
- `gemma4_vision_constant_subgraphs.mojo` — about 12 KB, the constant
  subgraphs, emitted separately.

### 2. Compile it with both timing flags

```bash
source ./utils/start-modular.sh          # once per shell, puts mojo on PATH

MODULAR_CACHE_DIR=$(mktemp -d) \
mojo build --emit=object --mlir-timing --llvm-timing \
  --target-accelerator=sm_100a gemma4_vision+gemma4_language.mojo \
  -o /dev/null 2> log.txt
```

`--target-accelerator` selects the accelerator to generate code for and needs
no GPU present, only a valid arch such as `sm_100a` or `gfx950`. `-o /dev/null`
skips writing the object file, since the timing is the point.

Record which build of `mojo` produced the log. A debug build and a production
build are not comparable — see
[Notes for benchmarking](#notes-for-benchmarking).

## Digesting the report

The raw log for a model is large; gemma-4 produces about 165,000 lines, most of
it one row per pass per accelerator module. The `mojo-compile-timing` skill
digests it:

```text
/mojo-compile-timing log.txt --top 10
```

Or run its script directly:

```bash
python3 .claude/skills/mojo-compile-timing/scripts/digest_timing_log.py log.txt
```

Useful options:

- `--compare BASELINE` — a second log alongside, with per row speedups. The
  first argument is the subject, the second the baseline.
- `--top N` — passes to name per group before rolling the tail up (default 5).
- `--markdown` — the tree in a fenced block plus a rollup table.
- `--json` — machine readable, for a dashboard.

The output for the gemma-4 debug compile above:

```text
MLIR root = whole compile                              265.5s (100.0%)
│
├─ ElaborateGenerators                                 119.7s ( 45.1%)
│  ├─ offload nvptx64 sm_100a scope "(also in host)"   101.9s ( 38.4%)
│  │  ├─ MLIR passes                                    57.2s ( 21.5%)
│  │  ├─ LLVM                                           23.0s (  8.6%)
│  │  └─ translation to LLVM IR, object emit            21.7s (  8.2%)
│  └─ elaboration itself                                17.9s (  6.7%)
│
├─ other host MLIR passes                              102.2s ( 38.5%)
│
└─ Rest = host code generation                          42.8s ( 16.1%)
   ├─ LLVM                                              34.5s ( 13.0%)
   └─ translation to LLVM IR, object emit                8.4s (  3.1%)

Rolled up by compiler half
  MLIR                                                 177.3s ( 66.8%)
  LLVM                                                  57.4s ( 21.6%)
  translation, object emit, untimed                     30.1s ( 11.3%)

Accelerator modules compiled: ~765
```

Every percentage is a share of the whole compile, so rows at any depth compare
directly. A child is part of its parent, never additional to it, so only rows
at the same indent sum to their parent.

The headline for this model: `ElaborateGenerators` looks like the hot spot at
45% of the compile, but only 17.9s of its 119.7s is elaboration. The other
101.9s is a complete nested compile of the accelerator code, driven from the
elaborator through `CompileOffloadOp`.

## Reading the numbers by hand

Four things the numbers do not say directly. The digest script handles all of
them; they matter when checking its output or writing another consumer.

### The MLIR root is the whole compile

It spans parsing, the passes, and code generation, because the `MLIRPassTiming`
object lives for all of `build()` in `KGEN/tools/mojo/Build/mojo-build.cpp`. On
one gemma-4 compile the root read 272.34s against 272.56s of wall clock. Take
the compile total from the root; never add sections to it.

### Each accelerator scope is counted twice

It prints at the outermost level next to `ElaborateGenerators`, but its time is
also inside it, which is what the `(also in host)` tag means. On that same log,
summing the outermost rows gives 323.80s against a true root of 265.53s.

That surplus is why the `Rest` row in the excerpt above is negative. MLIR
computes `Rest` as the root minus the sum of the children, so double counted
time lands there with its sign flipped: the row reads `-59.0020` in place of the
42.85s that is genuinely unattributed. Adding the scope back recovers it:
`-59.00 + 101.85 = 42.85` seconds. A consumer of the report has to either skip
rows whose name starts with `===---` or subtract them.

### `Rest` is host code generation, not noise

It is the untimed tail after the pass pipeline. `compileModuleToArchive` runs
`runKGENPipeline` with the timing scope attached, then hands off to the object
compiler, which has none. For gemma-4 the 42.8s splits into 34.5s of host LLVM
and 8.4s of translation to LLVM IR plus object emission.

### The LLVM groups overlap

Each LLVM section prints up to four groups, and only two are disjoint:

| Group                                  | Relationship                       |
|----------------------------------------|------------------------------------|
| `Pass execution timing report`         | the passes                         |
| `Analysis execution timing report`     | the analyses, separate from passes |
| `Instruction Selection and Scheduling` | sub-timers inside the ISel pass    |
| `Register Allocation`                  | sub-timers inside the RA pass      |

The last two come from `NamedRegionTimer` objects inside
`SelectionDAGISel::CodeGenAndEmitDAG` and the greedy allocator, so their time is
already counted by the pass report. LLVM time for a target is
`pass execution + analyses`. Use the other two groups only as a breakdown of
their parent pass: of the 8.9s in `AArch64 Instruction Selection`, 3.3s is
`DAG Combining after legalize types`.

One more detail when ranking accelerator passes: the report has one row per pass
per module, so `InstCombinePass` appears 765 times at about 0.30s each.
Aggregate by name, stripping the `#N` instance suffix, or the top of the list
is ten identical rows.

## The tracked benchmark

The steps above time one compile on one machine. To watch compile time move
over weeks, CI runs the same measurement daily through
`utils/benchmarking/kepler/mojo_compilation/`, which has two targets:

| Target                     | What it does                                                 |
|----------------------------|--------------------------------------------------------------|
| `make_snapshot`            | Builds the frozen input and publishes it to S3.              |
| `bench_mojo_compile_time`  | Compiles that input and reports the breakdown.               |

The input is frozen because a compile time only means something against a fixed
input. The emitted Mojo changes whenever the graph compiler or the kernel
library changes, and either can move compile time as much as the Mojo compiler
can. A snapshot pins all of it, so a step in the series is attributable.

The snapshot holds the library as **source**, not as precompiled `.mojoc`. A
`.mojoc` loads only in a compiler whose MLIR dialect checksum matches, and that
checksum hashes the dialect `.td` text, so it changes most days — a snapshot of
compiled packages is unusable within a day of being taken. Source stays
compilable for as long as the language accepts it, so the benchmark rebuilds the
packages with the compiler under test on every invocation. That rebuild is not
overhead: it compiles the whole standard library and every kernel library, far
more code than one emitted model, and is reported as `precompile.*`.

### Generating a snapshot

```bash
./bazelw run //utils/benchmarking/kepler/mojo_compilation:make_snapshot -- \
  --out ~/mojo-compile-snapshot --skip-upload
```

`--skip-upload` builds and archives the snapshot without publishing it, which is
what a local run wants. It also bypasses the cadence check, so the snapshot is
always rebuilt.

The run emits the Mojo for every model in `model_inputs.yaml` the same way
[step 1 above](#1-get-the-emitted-mojo) does, copies the library sources, reads
the precompile recipe out of `bazel aquery`, rebuilds the packages once to
confirm the frozen tree compiles, and counts each source's offload kernels. It
produces:

```text
~/mojo-compile-snapshot/
├── SNAPSHOT.yaml               the recipe, plus a hash and size per file
├── library/                    stdlib, kernel and max sources at their repo paths
├── sources/<dir>/              the emitted .mojo files
└── mojo_compile_snapshot.tar.gz
```

Two things to expect. The graph compiler run needs HuggingFace access to the
model and fails on the target it cannot reach from here, exactly as in step 1;
its output is kept in a log and only shown if a source is missing, which is the
real failure. And `--skip-emit` reuses the `.mojo` already under `--out` instead
of running the graph compiler again, which is much faster when iterating on
everything downstream of the emit.

### Running the benchmark locally

```bash
./bazelw run //utils/benchmarking/kepler/mojo_compilation:bench_mojo_compile_time -- \
  --snapshot ~/mojo-compile-snapshot \
  --runs 1 \
  --results /tmp/compile-time.json
```

| Option        | Meaning                                                             |
|---------------|---------------------------------------------------------------------|
| `--list`      | Print the benchmark names the manifest declares, and exit.          |
| `--snapshot`  | Compile an unpacked snapshot instead of fetching the published one. |
| `--runs`      | Repeats per source, each two full compiles (default: 3).            |
| `--results`   | Also write the results to a file; `.json` or `.yaml` by extension.  |
| `--benchmark` | Run only the named benchmark. Repeatable.                           |

The first four are this benchmark's own. `--benchmark` comes from the shared
Kepler runner, as does everything else on that command line not listed here.

To see what there is to measure:

```bash
./bazelw run //utils/benchmarking/kepler/mojo_compilation:bench_mojo_compile_time \
  -- --list
```

```text
gemma-4-31b.language            sm_100a  gemma4/gemma4_vision+gemma4_language.mojo
gemma-4-31b.constant-subgraphs  sm_100a  gemma4/gemma4_vision_constant_subgraphs.mojo
```

Then measure one of them, naming it exactly as `--list` printed it:

```bash
./bazelw run //utils/benchmarking/kepler/mojo_compilation:bench_mojo_compile_time -- \
  --snapshot ~/mojo-compile-snapshot \
  --runs 1 \
  --benchmark gemma-4-31b.language
```

The library rebuild happens either way: it is the input to whatever is being
measured, so filtering saves the model compiles and nothing else.

Each repeat compiles the source twice: once with no timing flags at the
compiler's default thread count, reported as `wall.total`, and once with
`--mlir-timing --llvm-timing` for the phase breakdown. Only the first is a
latency figure, for the reason in [the flags](#the-flags) — the second is
single threaded.

Omitting `--snapshot` fetches the most recently published snapshot over HTTPS,
which is what CI does.

Budget for the cost in compiles rather than in minutes, which vary by machine.
An invocation is one library rebuild plus, per source, two full compiles per
repeat. The rebuild happens once per invocation and not once per repeat, since
every source compiles against the same packages, and the large emitted source
dominates everything else. Start with `--runs 1`.

### Adding a model

Every series the benchmark tracks comes from `model_inputs.yaml` beside the two
targets. A model there is one extraction — one run of the graph compiler — and
the files that extraction produces are its sources, each compiled and plotted on
its own.

To add one:

1. Emit it by hand first, with the command from
   [step 1 above](#1-get-the-emitted-mojo) pointed at the new model, and look at
   what lands in the output directory. You need the emitted file names, and they
   are not predictable from the model name.
2. Add an entry:

   ```yaml
   - name: my-model
     target_accelerator: sm_100a
     emitted_by:
       model_path: org/My-Model
       target: cuda:sm_100a
       command: >-
         MODULAR_DEBUG=ir-output-dir=<out-dir>
         ./bazelw run //max/python/max/_entrypoints:pipelines -- warm-cache
         --model-path org/My-Model --target cuda:sm_100a
     sources:
       - name: language
         file: my-model/the_emitted_file.mojo
   ```

3. Regenerate the snapshot. The emit, the offload-kernel count and the source
   statistics all follow from the manifest, so nothing else needs editing.

Four things to get right:

- `file` is the path **inside the snapshot**, not where the extraction wrote it.
  Only its basename has to match an emitted file; the leading directory is
  yours to choose, and exists to keep one model's sources away from another's.
- `emitted_by` is not read at benchmark time. It is recorded so that a refresh
  is reproducible, since what the graph compiler emits depends on those
  settings — so keep `command` in step with the fields above it.
- `name` and each source's `name` form the series name, `<model>.<source>`.
  They are the dashboard's identity for the series: renaming one starts a fresh
  line rather than continuing the existing one. A model name must not contain a
  dot, because the BigQuery view splits the series name on the first one.
- Each source costs two full compiles per repeat on every CI run, so add
  sources that answer a question, not every file an extraction happens to
  produce.

The results land in BigQuery and Looker. See the Kepler docs for the tables and
the dashboard: `docs/internal/KeplerBenchmarking.md`.

## Notes for benchmarking

Benchmark the production build. It is what ships, and it is faster, so runs cost
less. More importantly the two builds are not comparable: on the gemma-4 pair
measured here, production ran 56 outermost passes against 60 in debug, with
`LowerGlobalPOPToLLVM` absent entirely, and `VerifyParameters` ran twice instead
of five times. Per pass speedups ranged from 1.2x to 8.2x, so a debug
measurement cannot be scaled into a production estimate.

Because the timing flags force single threaded compilation, these numbers are
total CPU work rather than the latency a user sees. Tracking user visible
compile time needs a second measurement with default threads and no timing
flags.

The accelerator module count is a property of the input, not the compiler. If it
moves, the emitted Mojo changed shape, so compare compile times across snapshots
only while it holds steady.

Some time still belongs to no pass, about 10% of the compiles measured so far:
the translation and object emit rows on both the host and the accelerator side.
A regression landing there shows up only in the total.
