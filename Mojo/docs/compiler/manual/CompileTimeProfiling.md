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
| `--timing-json`              | Writes the reports as one JSON object instead of as text.                                      |
| `--timing-file FILE`         | Writes the reports to `FILE` instead of to stderr.                                             |

See them and their full help text with `mojo build --help-hidden` or
`mojo run --help-hidden`. Their definitions live in
`Mojo/tools/mojo/Common/CompilationOptions.td`.

Four things to know before reading any report they produce:

1. Both reports go to **stderr** when compilation finishes, so capture with
   `2>`. For `mojo run` they print before the program starts. `--timing-file`
   redirects them to a file instead, in either format; see
   [JSON output](#json-output).
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

## JSON output

`--timing-json` writes the reports as one JSON object rather than as text, and
`--timing-file FILE` sends them to a file rather than to stderr. The two are
independent, so JSON on stderr and text in a file both work. Neither changes
the measurement: `--llvm-timing` still forces single threaded compilation, and
a warm cache still reports nothing.

The object holds one member per report, and only the members the command asked
for — `--timing-json` with no timing flag gets `{}`.

```json
{
  "mlir": [
    {
      "name": "ElaborateGenerators",
      "wall": {"duration": 119.7262, "percentage": 45.1},
      "passes": [
        {
          "name": "offload nvptx64-nvidia-cuda sm_100a (also in host)",
          "wall": {"duration": 101.8517, "percentage": 38.4},
          "passes": []
        }
      ]
    }
  ],
  "llvm": [
    {
      "pipeline": "offload nvptx64-nvidia-cuda sm_100a",
      "times": {"time.pass.SROAPass.wall": 0.000464}
    }
  ]
}
```

`mlir` follows `--mlir-timing-display`. `tree` nests each entry's children
under `passes`; `list` is flat and opens with a `root` entry. `llvm` is one
entry per pipeline in the order they ran — each accelerator target, then the
host — each with a flat `times` map.

Those `times` keys are `time.<group>.<name>.<metric>`, with `<metric>` one of
`wall`, `user`, `sys` or `instr`. The four groups are the same four the text
report prints and they overlap the same way, so the arithmetic in
[The LLVM groups overlap](#the-llvm-groups-overlap) applies unchanged.

One difference from the text report: accelerator scope names in `mlir` carry no
`===---` rule. The rule exists to set the scope apart from the host passes it
sits beside when a human reads a tree, and it only gets in a matcher's way.
Match the `(also in host)` suffix instead.

The digest script reads the text report, not the JSON. Use `--timing-json`
when feeding another consumer, and `2> log.txt` or `--timing-file` when feeding
[the digest](#digesting-the-report).

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
object lives for all of `build()` in `Mojo/tools/mojo/Build/mojo-build.cpp`. On
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
those scope rows or subtract them — by the `===---` prefix in the text report,
or by the `(also in host)` suffix, which both formats carry.

### `Rest` is host code generation, not noise

It is the untimed tail after the pass pipeline. `compileModuleToArchive` runs
`runKGENPipeline` with the timing scope attached, then hands off to the object
compiler, which has none. For gemma-4 the 42.8s splits into 34.5s of host LLVM
and 8.4s of translation to LLVM IR plus object emission.

### The LLVM groups overlap

Each LLVM section prints up to four groups, and only two are disjoint:

| Group                                  | JSON key prefix    | Relationship                       |
|----------------------------------------|--------------------|------------------------------------|
| `Pass execution timing report`         | `time.pass.`       | the passes                         |
| `Analysis execution timing report`     | `time.analysis.`   | the analyses, separate from passes |
| `Instruction Selection and Scheduling` | `time.sdag.`       | sub-timers inside the ISel pass    |
| `Register Allocation`                  | `time.regalloc.`   | sub-timers inside the RA pass      |

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

The steps above time one compile on one machine. To watch compile time move over
weeks, the `Mojo Compile Time` workflow runs the same measurement daily through
`utils/benchmarking/kepler/mojo_compilation/`, which has two targets:

| Target                    | What it does                                            |
|---------------------------|---------------------------------------------------------|
| `make_artifacts`          | Runs the graph compiler and keeps the Mojo it emits.    |
| `bench_mojo_compile_time` | Compiles that Mojo and reports the pipeline breakdown.  |

Both run at one commit, so the graph compiler, the kernel library and the Mojo
compiler all move together. A step in the series therefore says compile time
moved, not what moved it — which is bisected like any other regression. The
`input.*` series record what the compiler was given, so a step that comes from
the graph compiler emitting more code is visible beside the timings.

They are separate targets because emitting is expensive: it builds the model's
whole graph from its checkpoint. Locally you pay that once and then compile the
result as often as you like.

Two models are tracked. `gemma-4-31b` is the small end; `kimi-k25`
(`nvidia/Kimi-K2.5-NVFP4`, sharded over eight devices) is the large one, and
about two thirds of the daily job. `kimi-k25` sits out the per-PR comparison —
see [comparing a pull request](#comparing-a-pull-request) — because emitting
and compiling it takes around 35 minutes, which two arms and a repeat would not
fit.

### Emitting the sources

```bash
./bazelw run //utils/benchmarking/kepler/mojo_compilation:make_artifacts -- \
  --out ~/mojo-artifacts
```

| Option      | Meaning                                                     |
|-------------|-------------------------------------------------------------|
| `--out`     | Directory to emit into; created if absent.                  |
| `--inputs`  | Model manifest (default: `model_inputs.yaml`).              |
| `--model`   | Emit one model by name, rather than every model.            |
| `--pr-only` | Emit only the models the per-PR comparison measures.        |

Each source lands at `<out>/<model>/<source>.mojo`, beside an `ARTIFACTS.yaml`
recording the commit, the name the graph compiler actually gave each file, and
what each file looks like. The graph compiler run fails on a target it cannot
reach from here; its output is kept in a log and only shown when a source is
missing, which is the real failure.

The emit needs the model's **whole checkpoint on local disk** — 58 GiB for
`gemma-4-31b`, 550 GiB for `kimi-k25` — and downloads every weight file it does
not already have. CI does not pay this: the benchmark runner mounts a shared
HuggingFace cache that already holds both, which is why `HF_HOME` points into
it. On a workstation the first emit of `kimi-k25` is a 550 GiB download, so
check what you already have before starting one — `HF_HOME` is usually unset
locally, in which case the cache is `~/.cache/huggingface/hub`:

```bash
du -sh "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--nvidia--Kimi-K2.5-NVFP4"
```

Measuring needs none of this. `bench_mojo_compile_time` reads only the `.mojo`
files, so once an emit has run — or once you have copied its output directory
from elsewhere — the compile loop is offline and the checkpoint is irrelevant.

### Measuring

```bash
./bazelw run //utils/benchmarking/kepler/mojo_compilation:bench_mojo_compile_time -- \
  --sources ~/mojo-artifacts \
  --runs 1 \
  --results /tmp/compile-time.json
```

| Option        | Meaning                                                             |
|---------------|---------------------------------------------------------------------|
| `--list`      | Print the benchmark names the manifest declares, and exit.          |
| `--sources`   | Directory `make_artifacts` emitted into.                            |
| `--runs`      | Repeats per source, each two full compiles (default: 3).            |
| `--results`   | Also write the results to a file; `.json` or `.yaml` by extension.  |
| `--benchmark` | Run only the named benchmark. Repeatable.                           |

The first four are this benchmark's own; `--benchmark` comes from the shared
Kepler runner, as does everything else on that command line. In CI the sources
directory is named by `MOJO_COMPILE_SOURCES` instead, because the benchmark runs
as a bazel test and cannot take arguments. `MOJO_COMPILE_PR_ONLY=1` restricts
the run to the models the per-PR comparison measures, matching
`make_artifacts --pr-only`; it is an environment variable because benchmarks are
registered before any argument is parsed.

Each repeat compiles the source twice: once with no timing flags at the
compiler's default thread count, reported as `wall.total`, and once with
`--mlir-timing --llvm-timing` for the phase breakdown. Only the first is a
latency figure, for the reason in [the flags](#the-flags) — the second is single
threaded.

Budget for the cost in compiles rather than in minutes, which vary by machine:
per source, two full compiles per repeat, and the large emitted source dominates
everything else. Start with `--runs 1`.

#### One model at a time

Emitting and compiling every model takes the better part of an hour, and each
emit wants its checkpoint on disk, so a local loop usually wants one model.
`--model` restricts the emit and `--benchmark` the measurement; the series names
come from the manifest, so every model has them:

```bash
./bazelw run //utils/benchmarking/kepler/mojo_compilation:make_artifacts -- \
  --model kimi-k25 --out ~/mojo-artifacts

./bazelw run //utils/benchmarking/kepler/mojo_compilation:bench_mojo_compile_time -- \
  --sources ~/mojo-artifacts --runs 1 \
  --benchmark kimi-k25.language
```

`--list` prints the names to choose from. Note that `--benchmark` reaches the
binary only under `bazelw run`; through `bazelw test` it has to be passed as
`--test_arg=--benchmark --test_arg=<name>`.

### Adding a model

Every series comes from `model_inputs.yaml` beside the two targets. A model
there is one extraction — one run of the graph compiler — and the files it
produces are its sources, each compiled and plotted on its own.

To add one:

1. Add the entry with a `match` you expect to be right:

   ```yaml
   - name: my-model
     target_accelerator: sm_100a
     emitted_by:
       model_path: org/My-Model
       target: cuda:sm_100a
       extra_args:
         - --devices
         - gpu:0,1
       command: >-
         MODULAR_DEBUG=ir-output-dir=<out-dir>
         ./bazelw run //max/python/max/_entrypoints:pipelines -- warm-cache
         --model-path org/My-Model --target cuda:sm_100a --devices gpu:0,1
     sources:
       - name: language
         match: "*_language.mojo"
   ```

2. Emit it: `make_artifacts --model my-model --out <dir>`. If a `match` is
   wrong, the error lists every `.mojo` the graph compiler emitted, which is
   what you need to correct it — the names are not predictable from the model
   name, so guessing first and reading the error is the short path.
3. Re-emit. The statistics and the series follow from the manifest, so nothing
   else needs editing.

Four things to get right:

- `match` is a glob and has to match exactly one emitted file. Prefer one that
  keys off the part of the name you care about — the graph compiler builds a
  filename by joining the graphs it holds with `+`, so a full name is hostage to
  how the graph happens to be partitioned. Zero matches and several matches are
  both errors; neither silently measures the wrong file.
- `emitted_by` is what the emitter runs, so keep `command` in step with the
  fields above it. `extra_args` is for anything past `--model-path` and
  `--target`: a sharded model needs its device list, and an MoE its
  `--ep-size`, or the emitted graph is not the one that gets served.
- `name` and each source's `name` form the series name, `<model>.<source>`.
  They are the dashboard's identity for the series: renaming one starts a fresh
  line rather than continuing the existing one. A model name must not contain a
  dot. The source name also names the stored file, `<model>/<source>.mojo`.
- Set `in_pr_comparison: false` if the model is too slow for the per-PR
  comparison to carry. The daily benchmark still measures it.

### Reading the results

Every run uploads to BigQuery, and the series are plotted at
<https://benchmark-visibility.prod.modular-internal.com/perf/mojo-compile-time>.
See `docs/internal/KeplerBenchmarking.md` for the table itself.

The page draws three charts per source, because the series do not share an
axis: `wall.total` on its own, the pipeline phases from the flagged compile,
and the `input.*` counts. Points come from main only, one per CI run, and each
links to the commit it measured.

### Comparing a pull request

The daily series says compile time moved; to ask whether your branch moved it,
label the pull request `ci-mojo-compile-time-benchmark`. The `Mojo Compile Time
A/B` workflow then measures the two *arms* — the merge base and the branch
head — on one runner, and posts the comparison as a comment, refreshed on every
push while the label is on. The benchmark takes about an hour, and needs the
machine to itself for the timings to be stable, so it is label-gated rather
than run on every pull request.

Each arm builds its own compiler and emits its own Mojo, so a graph-compiler or
kernel-library change counts the same way a Mojo-compiler change does. The arms
are interleaved rather than run back to back, so drift over the run lands on
both of them, and a difference smaller than the spread between one arm's own
repeats is reported as noise rather than as a number to act on. The comment
uses the same word.

A regression is reported, never enforced: more compile time is often the honest
cost of more work. The check fails only when an arm fails to build or compile.

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
