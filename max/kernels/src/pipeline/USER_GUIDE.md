# Pipeline framework user guide

A practical guide for kernel developers who want to use the pipeline
scheduling framework. Includes two worked examples (simple and advanced) and
a sketch of how a future NVIDIA pipeline might look.

## Quick start

The framework lives in `max/kernels/src/pipeline/`. Import from submodules
directly:

```mojo
from pipeline.types import OpDesc, ResourceKind, OpRole, ScheduleOps
from pipeline.config import PipelineConfig, ScheduleConfig, TargetProfile
from pipeline.compiler import PipelineSchedule, compile_schedule
```

To define a pipeline schedule, implement the `PipelineSchedule` trait with
two required methods. The framework derives everything else.

## Core concepts

### Operations and the loop body

A pipeline loop body is a list of `OpDesc` values. Each `OpDesc` describes
one operation in the loop iteration:

```mojo
var op = OpDesc.op(
    MY_LOAD_A,           # tag: kernel-defined enum value (Int)
    ResourceKind.GLOBAL_MEM,  # resource: which hardware unit
    200,                      # latency: estimated cycles
    OpRole.GLOBAL_LOAD,       # role: scheduling role
    channel=0,                # buffer channel (A=0, B=1)
    stage=0,                  # LDS buffer stage
    subtile=0,                # K-subtile index
)
```

The four key fields:

- **`tag`**: identifies the operation. You define these per-kernel (see
  "Op tags" below).
- **`resource`**: which hardware unit runs this op. Determines which
  hardware counter tracks it (`vmcnt` for `GLOBAL_MEM`, `lgkmcnt` for `LDS`).
- **`role`**: the scheduling category. `GLOBAL_LOAD` means DRAM-to-LDS,
  `FRAGMENT_LOAD` means LDS-to-register, `COMPUTE` means MMA.
- **`latency`**: estimated cycles. Used by the scheduler to evaluate
  orderings. Doesn't need to be exact — it's a heuristic.

### Op tags

Each kernel defines its own op tags by implementing the `ScheduleOps` trait.
The framework reserves tags 128+ for infrastructure ops (barriers, fences,
waits). Your kernel ops use tags 0-127:

```mojo
@fieldwise_init
struct MyOps(ScheduleOps):
    var value: Int
    comptime LOAD_A = Self(0)
    comptime LOAD_B = Self(1)
    comptime MMA_LOAD_A = Self(2)
    comptime MMA_LOAD_B = Self(3)
    comptime MMA = Self(4)
```

### The PipelineSchedule trait

```mojo
trait PipelineSchedule:
    # Required: pipeline structure (depth, MMA grid, etc.)
    def config(self) -> PipelineConfig:
        ...

    # Required: the pipelined loop body as a list of operations
    def build_body(self) -> List[OpDesc]:
        ...

    # Optional: dependency edges (default: inferred from ops + config)
    def derive_edges(self, body: List[OpDesc]) -> List[DepEdge]:
        ...

    # Optional: tuning knobs (default: ScheduleConfig())
    def schedule_config(self) -> ScheduleConfig:
        ...

    # Optional: post-process kernel entries (default: identity)
    def transform_kernel(
        self, ker: List[ScheduleEntry], body: List[OpDesc]
    ) -> List[ScheduleEntry]:
        ...
```

You implement `config()` and `build_body()`. The framework handles the
rest: dependency analysis, scheduling, wait-count derivation, and
prologue/epilogue generation.

### Compilation and consumption

```mojo
# Compile at comptime
comptime schedule = compile_schedule(MySchedule(...))

# Consume in the kernel
comptime for i in range(len(schedule.prologue)):
    dispatch[schedule.prologue[i]]()

for k in range(K_start, K_end, BK):
    comptime for i in range(len(schedule.kernel)):
        dispatch[schedule.kernel[i]]()

comptime for i in range(len(schedule.epilogue)):
    dispatch[schedule.epilogue[i]]()
```

The `comptime for` unrolls at compile time. Each `ScheduleEntry` carries an
`OpDesc` that your dispatch function maps to actual hardware calls.

### The dispatch function

You write a `@__parameter` function that maps `ScheduleEntry` tags to kernel
primitives:

```mojo
@__parameter
@always_inline
def dispatch[entry: ScheduleEntry]():
    comptime if entry.op.tag == MyOps.LOAD_A.value:
        load_a_from_dram(entry.op.stage)
    elif entry.op.tag == MyOps.MMA.value:
        mma_op.mma[entry.op.subtile]()
    elif entry.op.tag == MyOps.BARRIER.value:
        barrier()
    # ... etc
```

This compiles to straight-line code — no branching at runtime.

## The declarative body builder

Instead of constructing `OpDesc` values manually, use `PipelineBody` for a
concise declaration:

```mojo
from pipeline_body import PipelineBody

def _my_body() -> List[OpDesc]:
    with PipelineBody() as b:
        b.load(LOAD_A, ch=0, stage=0, sub=0)     # global load A
        b.load(LOAD_B, ch=1, stage=0, sub=0)     # global load B
        b.fan[2](MMA_LOAD_A, ch=0, stage=0)      # 2 fragment loads A
        b.fan[2](MMA_LOAD_B, ch=1, stage=0)      # 2 fragment loads B
        b.grid[2, 2](MMA)                        # 2x2 MMA grid
        return b.done()
```

- `load()`: one global load op
- `store()`: one shared memory store op
- `barrier()`: a synchronization barrier
- `fan[N]()`: N ops with auto-incrementing subtile (0, 1, ..., N-1)
- `grid[M, N]()`: M*N compute ops in a 2D grid pattern

These produce bare logical ops — no resource, latency, or role annotations.
You annotate them later with a cost model.

## The cost model and target profile

The framework separates WHAT ops exist from HOW the hardware executes them.
A `TargetCostModel` maps op tags to hardware costs:

```mojo
from pipeline.types import OpCost, TargetCostModel

def my_cost_model() -> TargetCostModel:
    return TargetCostModel(
        costs=List[OpCost](
            OpCost(LOAD_A, ResourceKind.GLOBAL_MEM, 200, OpRole.GLOBAL_LOAD),
            OpCost(LOAD_B, ResourceKind.GLOBAL_MEM, 200, OpRole.GLOBAL_LOAD),
            OpCost(MMA_LOAD_A, ResourceKind.LDS, 20, OpRole.FRAGMENT_LOAD),
            OpCost(MMA_LOAD_B, ResourceKind.LDS, 20, OpRole.FRAGMENT_LOAD),
            OpCost(MMA, ResourceKind.MMA_UNIT, 16, OpRole.COMPUTE),
        )
    )
```

`TargetProfile` bundles the cost model with pipeline configuration:

```mojo
from pipeline.config import TargetProfile

def my_target() -> TargetProfile:
    return TargetProfile(
        cost_model=my_cost_model(),
        pipeline=my_pipeline_config(),
    )
```

The `annotate_ops()` function stamps logical ops with costs from the model:

```mojo
from pipeline.types import annotate_ops

var logical = _my_body()                      # bare ops (no costs)
var annotated = annotate_ops(logical, costs)  # ops with resource/latency/role
```

This separation means the same algorithm declaration works across different
hardware targets — change the cost model, get a different schedule.

## Example 1: single-buffer matmul (simple)

The default AMD matmul uses a single-buffer pipeline with barrier-gated
read/write phases. This is the simplest case.

**File**: `amd_matmul_schedule.mojo` (~140 lines of schedule code)

### Op tags

Four bundled operations (A+B combined per op):

```mojo
@fieldwise_init
struct DefaultMatmulOps(ScheduleOps):
    var value: Int
    comptime LOAD_DRAM = Self(0)    # DRAM → LDS (A+B)
    comptime STORE_SMEM = Self(1)   # register → LDS write (A+B)
    comptime LOAD_FRAG = Self(2)    # LDS → register (A+B, per k-tile)
    comptime COMPUTE = Self(3)      # MMA (per k-tile)
```

### Logical body

```mojo
def _logical_body[num_k_tiles: Int]() -> List[OpDesc]:
    with PipelineBody() as b:
        b.load(LOAD_DRAM, ch=0)                   # 1 DRAM load
        b.store(STORE_SMEM, ch=0)                  # 1 LDS store
        b.barrier()                                # read/write gate
        b.fan[num_k_tiles](LOAD_FRAG, ch=0)       # T fragment loads
        b.fan[num_k_tiles](COMPUTE)                # T MMAs
        return b.done()
```

This is the *entire* algorithm description. Five lines. The framework derives
the pipeline from this.

### Trait implementation

```mojo
struct SingleBufferSchedule[num_k_tiles: Int](PipelineSchedule):
    var hints: AMDScheduleHints
    var _target: TargetProfile

    def config(self) -> PipelineConfig:
        return self._target.pipeline

    def build_body(self) -> List[OpDesc]:
        var logical = _logical_body[Self.num_k_tiles]()
        var annotated = annotate_ops(logical, self._target.cost_model)
        var body = single_buffer_reorder(annotated, self._target.pipeline)
        return optimize_within_barriers(body, self._target.pipeline)

    def transform_kernel(
        self, ker: List[ScheduleEntry], body: List[OpDesc]
    ) -> List[ScheduleEntry]:
        # Append AMD-specific schedule_group_barrier hints
        var result = ker.copy()
        append_amd_hints(result, body, self.config(), self.hints)
        return result^
```

`build_body()` is the key: declare logical ops, annotate with hardware costs,
reorder for the barrier structure, then optimize within each segment.

The optional `transform_kernel()` appends AMD-specific schedule hints.
These are target-specific, so they live in the kernel, not the framework.

### Kernel consumption

```mojo
comptime schedule = build_default_matmul_schedule[
    num_k_tiles=num_k_tiles, ...
]()

@__parameter
@always_inline
def _bind[entry: ScheduleEntry]():
    comptime if entry.op.tag == LOAD_DRAM:
        load_tiles_from_dram()
    elif entry.op.tag == STORE_SMEM:
        copy_tiles_to_smem()
    elif entry.op.tag == LOAD_FRAG:
        mma_op.load_tile_fragment[entry.op.subtile](a_tiles, b_tiles)
    elif entry.op.tag == COMPUTE:
        mma_op.mma[entry.op.subtile]()
    elif entry.op.tag == DefaultMatmulOps.BARRIER.value:
        barrier()
    # ... schedule hints ...

# Execute
comptime for i in range(len(schedule.prologue)):
    _bind[schedule.prologue[i]]()
for _ in range(2, K // BK):
    comptime for i in range(len(schedule.kernel)):
        _bind[schedule.kernel[i]]()
comptime for i in range(len(schedule.epilogue)):
    _bind[schedule.epilogue[i]]()
```

The framework derived the prologue (fill the pipeline), kernel (steady state),
and epilogue (drain) from the five-line logical body.

## Example 2: double-buffer ping-pong matmul (advanced)

The ping-pong matmul uses double-buffered LDS with two warp groups running
one MMA phase apart. This is significantly more complex.

**File**: `amd_ping_pong_schedule.mojo` (~150 lines of schedule code)

### Op tags

Five separate operations (A and B are independent ops):

```mojo
@fieldwise_init
struct PingPongOps(ScheduleOps):
    var value: Int
    comptime LOAD_A = Self(0)       # DRAM → LDS for A
    comptime LOAD_B = Self(1)       # DRAM → LDS for B
    comptime MMA_LOAD_A = Self(3)   # LDS → register for A
    comptime MMA_LOAD_B = Self(4)   # LDS → register for B
    comptime MMA = Self(5)          # MFMA compute
```

### Logical body

The algorithm has two partitions (for two LDS stages). Each half declares its
ops with explicit stage and K-offset metadata:

```mojo
def _logical_half[h: Int]() -> List[OpDesc]:
    comptime s = h            # this half's LDS stage
    comptime os = 1 - h       # other half's LDS stage
    comptime k_off = KOffsetKind.K1 if h == 1 else KOffsetKind.K0

    with PipelineBody() as b:
        # Global loads: DRAM → LDS (4 per half: 2A + 2B)
        b.load(LOAD_A, ch=0, stage=os, sub=1, k=k_special)  # completion load
        b.load(LOAD_A, ch=0, stage=s,  sub=0, k=k_off)
        b.load(LOAD_B, ch=1, stage=s,  sub=0, k=k_off)
        b.load(LOAD_B, ch=1, stage=s,  sub=1, k=k_off)
        # Fragment loads: LDS → registers (2A + 2B per half)
        b.fan[2](MMA_LOAD_A, ch=0, stage=s)
        b.fan[2](MMA_LOAD_B, ch=1, stage=s)
        # MMA compute: 2×2 tile grid (4 per half)
        b.grid[2, 2](MMA)
        return b.done()
```

The full body combines both partitions: 24 ops total (12 per partition).

### Three-step build_body

```mojo
def build_body(self) -> List[OpDesc]:
    var logical = self.declare_ops()              # 24 bare ops
    var annotated = annotate_ops(logical, costs)  # attach hw costs
    return double_buffer_reorder(annotated, cfg)  # execution layout
```

1. **`declare_ops()`**: pure algorithm. What ops exist.
2. **`annotate_ops()`**: target hardware. What each op costs.
3. **`double_buffer_reorder()`**: execution structure. Interleave into
   MMA-centered blocks for double-buffer ping-pong.

### What the framework derives

From these 24 annotated ops, the framework automatically:

- Builds a `LoopBody` dependency graph (FLOW + ANTI edges)
- Runs CSP optimal scheduling to find the minimum-makespan ordering
- Groups ops into 8 MMA-centered blocks (4 per half)
- Redistributes global loads across blocks for latency hiding
- Derives `vmcnt`/`lgkmcnt` wait counts per block
- Derives drain masks for the epilogue
- Generates prologue (pipeline fill), kernel (steady state), epilogue (drain)
- Verifies the schedule at compile time (6 structural checks)

The kernel author writes ~150 lines. The framework does the rest.

### Compile-time scheduling

The `ScheduleConfig` controls the scheduling strategy:

```mojo
def schedule_config(self) -> ScheduleConfig:
    return ScheduleConfig(
        scheduling=SchedulingStrategy.CSP,  # or GREEDY, MANUAL
        auto_waits=True,                    # derive wait counts from blocks
    )
```

- **`CSP`**: exhaustive backtracking search for provably optimal ordering
- **`GREEDY`**: priority-based list scheduler (faster, may not be optimal)
- **`MANUAL`**: use the declaration order (for hand-tuned schedules)

For the 24-op ping-pong graph, CSP explores ~200 orderings at compile time
and proves optimality via a lower bound check.

## Sketch: NVIDIA SM90 pipeline (future direction)

The framework currently targets AMD CDNA3. NVIDIA Hopper (SM90) uses a
different synchronization model. This section walks through the real SM90
MHA (Flash Attention 3) pipeline in our codebase and shows concretely how
the framework's abstractions map to it.

### The SM90 MHA pipeline as it exists today

The FA3 kernel in `mha_sm90.mojo` uses **warp specialization**: a dedicated
producer warp loads K and V tiles via TMA (Tensor Memory Accelerator) into
shared memory, while consumer warp groups run WGMMA to compute attention.

**Producer side** (from `mha_fa3_utils.mojo`, `produce_k()`):

```mojo
# PipelineState tracks the circular buffer: index + phase bit
var state = PipelineState[pipeline_stages]()

def produce_k[wait: Bool](
    mut state: PipelineState[pipeline_stages],
    row: UInt32, kv_head_idx: UInt32,
):
    var write_idx = state.index()
    var write_phase = state.phase()

    # Wait for consumer to free this buffer stage
    comptime if wait:
        consumed_mbar_kv[write_idx].wait(write_phase)
        produced_mbar_kv[write_idx].expect_bytes(tile_bytes)

    # Async TMA copy: DRAM → shared memory, barrier tracks completion
    k_tma_op.async_copy(k_tile(write_idx), produced_mbar_kv[write_idx], coord)
    state.step()  # advance circular buffer
```

**Consumer side** (from `mha_sm90.mojo`, `q_mul_k()` and `p_mul_v()`):

```mojo
def q_mul_k(read_idx: UInt32, read_phase: UInt32, q_idx: UInt32):
    # Wait for producer to fill this K tile
    produced_mbar_kv[read_idx].wait(read_phase)

    # WGMMA: Q × K^T → P (attention scores)
    # Inputs from shared memory, accumulator in registers
    warpgroup_fence(p_reg_tile)
    wgmma_0.arrive()
    wgmma_0.wgmma[scale_c=0, ...](q_smem, k_smem, p_reg_tile, ...)
    wgmma_0.commit_group()       # close async MMA group
    warpgroup_fence(p_reg_tile)

def p_mul_v(read_idx: UInt32, read_phase: UInt32):
    # Wait for producer to fill this V tile
    produced_mbar_kv[read_idx].wait(read_phase)

    # WGMMA: P × V → O (output accumulation)
    # P from registers (softmax'd), V from shared memory
    warpgroup_fence(output_reg_tile)
    wgmma_1.arrive()
    wgmma_1.wgmma(p_frag, v_smem, output_reg_tile)
    wgmma_1.commit_group()
    warpgroup_fence(output_reg_tile)
```

WGMMA is **async and register-based on the accumulator side,
shared-memory-based on the input side**. The `commit_group()` /
`wait_group_sync[N]()` protocol counts pending MMA groups — semantically
identical to AMD's `vmcnt` counting pending memory ops.

### The inner software pipeline (the hard part)

The outer producer/consumer pipeline is relatively straightforward — one
warp loads, another computes. The real scheduling complexity is **inside
the consumer loop**, where two WGMMA groups overlap with softmax:

```mojo
# From mha_sm90.mojo, lines ~1727-1831 (steady-state inner loop)
# Each iteration processes K[i] and V[i-1]:

q_mul_k(K[i])             # issue Q×K[i] async — can't touch p_reg_tile
p_mul_v(V[i-1])           # issue P[i-1]×V[i-1] async — can't touch output
wait_for_q_mul_k[1](K[i]) # wait with 1 group in flight (P×V still running)
                           # → NOW safe to read p_reg_tile (Q×K result)
apply_mask(...)            # mask attention scores
softmax(p_reg_tile)        # online softmax on current Q×K scores
                           # (runs WHILE P[i-1]×V[i-1] is still executing)
wait_for_p_mul_v(V[i-1])  # wait for P×V to complete
                           # → NOW safe to modify output_reg_tile
scale_output(correction)   # rescale output accumulator with new softmax stats
```

This is a **depth-2 register-level software pipeline**:

```text
Iteration i:
  [Q×K[i] issued]──[P×V[i-1] issued]──[wait QK, 1 in flight]──[softmax]──[wait PV]──[scale]
                                         ↑                        ↑           ↑
                                     PV still running        overlapped     PV done
```

The `wait_group[1]` is the critical scheduling decision: it means "wait
until at most 1 WGMMA group remains in flight." Since Q×K was committed
first and P×V second, waiting for 1 group in flight means Q×K is done but
P×V may still be running. This lets the softmax computation overlap with
P×V execution — hiding the softmax latency behind the MMA.

**Why this is hard to reason about manually:**

- The correctness depends on the *ordering* of `commit_group()` calls:
  Q×K is committed first, so `wait_group[1]` retires Q×K, not P×V
- The `p_frag` (softmax'd scores) used by `p_mul_v(V[i-1])` was computed
  in iteration i-1 — it's a **loop-carried** value, exactly the pattern
  the framework's `DepEdge` with `loop_distance=1` models
- The register liveness is implicit: between `q_mul_k` and
  `wait_for_q_mul_k`, `p_reg_tile` is "owned" by the WGMMA unit and
  reading it would be undefined behavior
- The `consumed_mbar_kv[idx].arrive()` inside `wait_for_q_mul_k` signals
  the producer that the K buffer is free — so the barrier protocol is
  interleaved with the compute pipeline

This is exactly the kind of schedule that should be expressed declaratively
and verified at compile time, rather than maintained as carefully ordered
imperative code with comments like "can't rw `p_reg_tile`."

### How the framework would express this

The inner pipeline has **7 logical operations** per iteration, with
dependencies that determine the ordering:

```text
q_mul_k(K[i])          WGMMA group 0: Q×K → p_reg_tile
p_mul_v(V[i-1])        WGMMA group 1: P×V → output  (loop-carried P)
wait_qk[1]             wait: retire group 0, keep group 1 in flight
arrive_k(K[i])         signal: K buffer free for producer
softmax                register compute: p_reg_tile → p_frag (for next iter)
wait_pv                wait: retire group 1
scale_output           register compute: rescale output accumulator
```

The dependency graph:

```text
         ┌── FLOW d=0 ──→ WAIT_QK ──→ ARRIVE_K
Q_MUL_K ─┤                   │
         └── FLOW d=0 ──→ SOFTMAX ──→ P_MUL_V (next iter, d=1)
                                          │
P_MUL_V ── FLOW d=0 ──→ WAIT_PV ──→ SCALE_OUTPUT
         ↑
   SOFTMAX (prev iter, d=1: loop-carried p_frag)
```

The key edges:

- `Q_MUL_K → WAIT_QK[1]`: can't read p_reg_tile until Q×K completes
- `WAIT_QK → SOFTMAX`: softmax needs the Q×K result in p_reg_tile
- `SOFTMAX → P_MUL_V (d=1)`: next iteration's P×V uses this iteration's
  softmax'd p_frag — loop-carried dependency
- `P_MUL_V → WAIT_PV`: can't read output until P×V completes
- `WAIT_PV → SCALE_OUTPUT`: rescaling needs the P×V result
- `Q_MUL_K → ARRIVE_K`: must wait for K read before signaling K free

The framework's CSP scheduler, operating on this graph, would discover
that `P_MUL_V` can be issued immediately after `Q_MUL_K` (no dependency
between them in the same iteration), and that `SOFTMAX` overlaps with
`P_MUL_V` (the `wait_group[1]` pattern). The wait derivation would
determine that the minimum wait after Q×K is `wait_group[1]` (one group
still in flight), not `wait_group[0]` — because the scheduler knows
P×V doesn't need to complete yet.

This is the same analysis the framework does for AMD: in the ping-pong
kernel, the `vmcnt` wait counts determine how many pending loads to wait
for before issuing the next MMA. Here, the `wait_group[N]` counts
determine how many pending WGMMA groups to wait for before reading the
result registers.

**Barrier setup** (from `mha_sm90.mojo`, line ~1139):

```mojo
# Layout: [kv_smem] [produced_mbar × stages] [consumed_mbar × stages] [q_mbar × 2]
produced_mbar_kv = (kv_smem + kv_smem_size).bitcast[SharedMemBarrier]()
consumed_mbar_kv = produced_mbar_kv + pipeline_stages

# Initialize: 1 producer thread, num_consumer_threads consumers
comptime for i in range(pipeline_stages):
    produced_mbar_kv[i].init(1)
    consumed_mbar_kv[i].init(Int32(num_consumer_threads))
```

Two barrier arrays form a **full/empty** protocol: `produced_mbar[i]`
signals "stage i has data," `consumed_mbar[i]` signals "stage i is free."
Phase bits toggle on wrap-around to distinguish old signals from new ones.

The `ProducerConsumerPipeline` in `structured_kernels/pipeline.mojo`
abstracts this into a clean API:

```mojo
# Producer — wait for consumer, produce, advance
with pipeline.produce() as stage:
    tma_op.async_copy(tile(stage.index()), stage.mbar(), coord)

# Consumer — wait for producer, consume, signal done
with pipeline.consume() as stage:
    wgmma(tile(stage.index()))
```

### Hardware differences that matter

| Aspect            | AMD CDNA3                         | NVIDIA SM90                               |
|-------------------|-----------------------------------|-------------------------------------------|
| Sync model        | Counters (`vmcnt`, `lgkmcnt`)     | Hardware barriers (`mbarrier`)            |
| Async loads       | `buffer_load_*_lds` (counted)     | TMA `cp.async_bulk_tensor` (barrier)      |
| MMA issue         | `v_mfma_*` (fire-and-forget)      | WGMMA `arrive`→`wgmma`→`commit_group`     |
| MMA completion    | Implicit (pipeline depth)         | `wait_group_sync[N]()` (N groups pending) |
| Pipeline tracking | Implicit in counter arithmetic    | Explicit `PipelineState` + phase bits     |
| Warp cooperation  | Stagger (same code, offset start) | Specialization (different code paths)     |

The deepest difference is **stagger vs specialization**. AMD's ping-pong
has all warps running the same schedule, offset by one phase — the framework
models this as a single interleaved program. SM90's producer/consumer warps
run fundamentally different code — the framework would need to emit two
separate programs from the same dependency graph.

### How the framework maps to SM90

Despite the hardware differences, the core abstractions translate directly.
Here's a concrete mapping from the FA3 pipeline:

**Step 1 — Op tags.** Each distinct async operation becomes an `OpDesc` tag:

```mojo
@fieldwise_init
struct FA3Ops(ScheduleOps):
    var value: Int

    # Producer ops (TMA loads)
    comptime TMA_LOAD_K = Self(0)    # DRAM → smem K tile
    comptime TMA_LOAD_V = Self(1)    # DRAM → smem V tile

    # Consumer ops (WGMMA + softmax)
    comptime WGMMA_QK = Self(2)      # Q × K^T → P (register accum)
    comptime SOFTMAX = Self(3)       # online softmax (register → register)
    comptime WGMMA_PV = Self(4)      # P × V → O (register accum)
```

This captures the five operations in the FA3 inner loop. Note that Q is
loaded once per tile row (outside the K/V loop), so it's not in the
steady-state body.

**Step 2 — Dependency graph.** The data flow is:

```text
TMA_LOAD_K ──FLOW──→ WGMMA_QK ──FLOW──→ SOFTMAX ──FLOW──→ WGMMA_PV
TMA_LOAD_V ──FLOW──────────────────────────────────────────→ WGMMA_PV
```

`WGMMA_QK` can't start until K is in shared memory (barrier wait).
`SOFTMAX` can't start until `WGMMA_QK` completes (`wait_group_sync`).
`WGMMA_PV` needs both the softmax'd P (registers) and V (shared memory).
`TMA_LOAD_V` can overlap with `WGMMA_QK` — no dependency between them.

The framework's `DepEdge` captures all of this. The ANTI edge (V buffer
reuse) is distance-1: the *next* iteration's `TMA_LOAD_V` can't start until
the *current* `WGMMA_PV` finishes reading V.

```mojo
# Framework infers these edges from op roles + stage metadata:
# FLOW d=0: TMA_LOAD_K → WGMMA_QK  (K tile ready)
# FLOW d=0: WGMMA_QK → SOFTMAX     (P scores ready)
# FLOW d=0: SOFTMAX → WGMMA_PV     (P normalized)
# FLOW d=0: TMA_LOAD_V → WGMMA_PV  (V tile ready)
# ANTI d=1: WGMMA_QK → TMA_LOAD_K  (K buffer reuse, next iter)
# ANTI d=1: WGMMA_PV → TMA_LOAD_V  (V buffer reuse, next iter)
```

**Step 3 — Cost model.** NVIDIA-specific latencies and resources:

```mojo
def sm90_cost_model() -> TargetCostModel:
    return TargetCostModel(costs=List[OpCost](
        # TMA loads: ~100 cycles, tracked by mbarrier (not vmcnt)
        OpCost(TMA_LOAD_K, ResourceKind.GLOBAL_MEM, 100, OpRole.GLOBAL_LOAD),
        OpCost(TMA_LOAD_V, ResourceKind.GLOBAL_MEM, 100, OpRole.GLOBAL_LOAD),
        # WGMMA: ~20 cycles per group, tracked by commit/wait groups
        OpCost(WGMMA_QK, ResourceKind.MMA_UNIT, 20, OpRole.COMPUTE),
        OpCost(WGMMA_PV, ResourceKind.MMA_UNIT, 20, OpRole.COMPUTE),
        # Softmax: register-only, overlaps with loads
        OpCost(SOFTMAX, ResourceKind.NONE, 10, OpRole.COMPUTE),
    ))
```

**Step 4 — Schedule.** The scheduler finds the optimal interleaving. For
FA3, the key scheduling insight is that `TMA_LOAD_V` overlaps with
`WGMMA_QK + SOFTMAX` — the V load starts as soon as K is issued, hiding
its latency behind the Q×K computation:

```text
Time →
Producer:  [TMA_K] [TMA_V]                [TMA_K'] [TMA_V'] ...
Consumer:          [wait_K] [QK] [softmax] [wait_V] [PV] [wait_K'] ...
                    ↑ mbar                  ↑ mbar        ↑ mbar
```

The framework's CSP scheduler would discover this overlap automatically
from the dependency graph — `TMA_LOAD_V` has no dependency on
`TMA_LOAD_K`, so it can be scheduled immediately after (or even in
parallel, given the TMA unit supports multiple in-flight loads).

**Step 5 — Wait derivation.** This is where the mapping is cleanest.
The framework already derives "wait for N pending operations" — the
concept is identical across both architectures:

| AMD                    | NVIDIA                       | Framework abstraction                     |
|------------------------|------------------------------|-------------------------------------------|
| `s_waitcnt vmcnt(N)`   | `mbarrier.wait(phase)`       | `OpDesc.wait_vm_n(N)` / barrier placement |
| `s_waitcnt lgkmcnt(N)` | `wgmma_wait_group_sync[N]()` | `OpDesc.wait_lgkm_n(N)` / group tracking  |

For NVIDIA, "wait for all TMA loads to stage S" becomes
`produced_mbar_kv[S].wait(phase)` — a barrier wait instead of a counter
wait. "Wait for N WGMMA groups to complete" is `wait_group_sync[N]()` —
semantically the same as `s_waitcnt lgkmcnt(N)`.

The framework's `derive_waits_from_blocks()` already computes the minimum
wait count per block. The extension for NVIDIA is a different *emission*
strategy: instead of emitting `s_waitcnt` instructions, emit
`mbarrier.wait()` calls and `wait_group_sync[N]()` calls. The analysis
that decides *where* and *how much* to wait is the same.

This directly applies to the inner WGMMA pipeline described above. The
`wait_group[1]` in `wait_for_q_mul_k` is a scheduling decision: "we
need Q×K's result, but P×V can keep running." The framework's wait
derivation would compute this automatically — it knows which ops
produce values consumed by the next op, and how many groups are in
flight at each point. Today this is encoded by hand in the FA3 kernel
with a carefully chosen template parameter `[wgmma_left_in_flight: Int]`.
The framework would derive it from the dependency graph.

### Expressing the inner WGMMA pipeline with the current framework

The inner consumer pipeline has exactly the structure the framework
already handles: a small set of operations with data dependencies, loop-
carried values, and async completion semantics. Here's how you'd express
it today, with no framework changes:

```mojo
@fieldwise_init
struct FA3InnerOps(ScheduleOps):
    var value: Int

    # Data operations
    comptime WGMMA_QK = Self(0)       # Q×K → p_reg_tile (async, group 0)
    comptime WGMMA_PV = Self(1)       # P×V → output (async, group 1)
    comptime SOFTMAX = Self(2)        # softmax(p_reg_tile) → p_frag
    comptime SCALE_OUTPUT = Self(3)   # rescale output accumulator
    comptime ARRIVE_K = Self(4)       # signal K buffer free to producer
    comptime APPLY_MASK = Self(5)     # mask attention scores
```

The loop body, with explicit dependency metadata:

```mojo
def _inner_body() -> List[OpDesc]:
    with PipelineBody() as b:
        # WGMMA Q×K: reads from smem (K stage), writes p_reg_tile
        # channel=0 (K buffer), stage tracks the smem circular buffer
        b.compute(WGMMA_QK, ch=0, stage=0)

        # WGMMA P×V: reads p_frag (register, loop-carried!) + smem (V stage)
        # channel=1 (V buffer)
        b.compute(WGMMA_PV, ch=1, stage=0)

        # Barrier arrive: signals producer that K buffer is free
        b.signal(ARRIVE_K, ch=0, stage=0)

        # Softmax: reads p_reg_tile, writes p_frag (for NEXT iteration's P×V)
        b.compute(SOFTMAX)

        # Apply mask: modifies p_reg_tile scores
        b.compute(APPLY_MASK)

        # Scale output: reads/modifies output_reg_tile with softmax correction
        b.compute(SCALE_OUTPUT)

        return b.done()
```

The cost model — WGMMA groups are the "expensive" async ops, everything
else is register compute:

```mojo
def sm90_inner_cost_model() -> TargetCostModel:
    return TargetCostModel(costs=List[OpCost](
        # Async WGMMA groups (~20 cycles each, tracked by commit/wait)
        OpCost(WGMMA_QK, ResourceKind.MMA_UNIT, 20, OpRole.COMPUTE),
        OpCost(WGMMA_PV, ResourceKind.MMA_UNIT, 20, OpRole.COMPUTE),
        # Register-only compute (overlaps freely with async ops)
        OpCost(SOFTMAX, ResourceKind.NONE, 8, OpRole.COMPUTE),
        OpCost(SCALE_OUTPUT, ResourceKind.NONE, 4, OpRole.COMPUTE),
        OpCost(APPLY_MASK, ResourceKind.NONE, 4, OpRole.COMPUTE),
        # Barrier signal (~0 cycles, just a message)
        OpCost(ARRIVE_K, ResourceKind.SCALAR, 0, OpRole.SIGNAL),
    ))
```

The dependency edges — these are the key correctness constraints:

```mojo
def sm90_inner_edges(body: List[OpDesc]) -> List[DepEdge]:
    var edges = List[DepEdge]()

    # Within this iteration (d=0):
    # Q×K must complete before softmax can read p_reg_tile
    edges.append(DepEdge(WGMMA_QK, APPLY_MASK, DepKind.FLOW, loop_distance=0))
    edges.append(DepEdge(APPLY_MASK, SOFTMAX, DepKind.FLOW, loop_distance=0))

    # Q×K must complete before we signal K buffer free
    edges.append(DepEdge(WGMMA_QK, ARRIVE_K, DepKind.FLOW, loop_distance=0))

    # P×V must complete before we can rescale output
    edges.append(DepEdge(WGMMA_PV, SCALE_OUTPUT, DepKind.FLOW, loop_distance=0))

    # Across iterations (d=1):
    # This iteration's softmax produces p_frag consumed by NEXT iter's P×V
    edges.append(DepEdge(SOFTMAX, WGMMA_PV, DepKind.FLOW, loop_distance=1))

    # This iter's SCALE_OUTPUT writes output_reg_tile; next iter's P×V reads it
    edges.append(DepEdge(SCALE_OUTPUT, WGMMA_PV, DepKind.FLOW, loop_distance=1))

    return edges^
```

The `PipelineSchedule` implementation:

```mojo
struct FA3InnerSchedule(PipelineSchedule):
    var _target: TargetProfile

    def config(self) -> PipelineConfig:
        return PipelineConfig(
            depth=1,          # single iteration in flight
            m_mmas=1,         # 1 "MMA block" per iteration
            n_mmas=1,
            num_partitions=1,
            mma_serial=True,  # WGMMA groups serialize
            mma_latency=20,
            # ...
        )

    def build_body(self) -> List[OpDesc]:
        var logical = _inner_body()
        return annotate_ops(logical, self._target.cost_model)

    def derive_edges(self, body: List[OpDesc]) -> List[DepEdge]:
        return sm90_inner_edges(body)
```

The compiled schedule would produce the same ordering the handwritten code
has, but **derived from the graph**:

```text
Iteration i:
  0: WGMMA_QK(K[i])          ← issue group 0
  1: WGMMA_PV(V[i-1])        ← issue group 1 (no dependency on QK in this iter)
  2: WAIT_GROUP[1]            ← framework-derived: QK done, PV still in flight
  3: ARRIVE_K(K[i])           ← signal K free (depends on QK completion)
  4: APPLY_MASK               ← mask scores (depends on QK completion)
  5: SOFTMAX                  ← produces p_frag for next iter's PV
  6: WAIT_GROUP[0]            ← framework-derived: PV done
  7: SCALE_OUTPUT             ← rescale (depends on PV completion)
```

The `WAIT_GROUP[1]` at step 2 is the scheduling insight that the framework
derives automatically: the dependency graph says SOFTMAX needs QK's result
but not PV's. With 2 groups in flight (QK=group 0, PV=group 1), waiting
for 1 group in flight retires group 0 (QK) while group 1 (PV) keeps
running. The framework's wait derivation computes this from the op ordering
and the MMA resource model — the same analysis it does for `vmcnt` on AMD.

The dispatch function maps to the existing WGMMA ceremony:

```mojo
@__parameter
@always_inline
def inner_dispatch[entry: ScheduleEntry]():
    comptime if entry.op.tag == FA3InnerOps.WGMMA_QK.value:
        warpgroup_fence(p_reg_tile)
        wgmma_0.arrive()
        wgmma_0.wgmma[...](q_smem, k_smem, p_reg_tile, ...)
        wgmma_0.commit_group()
        warpgroup_fence(p_reg_tile)
    elif entry.op.tag == FA3InnerOps.WGMMA_PV.value:
        warpgroup_fence(output_reg_tile)
        wgmma_1.arrive()
        wgmma_1.wgmma(p_frag, v_smem, output_reg_tile)
        wgmma_1.commit_group()
        warpgroup_fence(output_reg_tile)
    elif entry.op.tag == FA3InnerOps.SOFTMAX.value:
        online_softmax(p_reg_tile, p_frag, rowmax, rowsum)
    elif entry.op.tag == FA3InnerOps.SCALE_OUTPUT.value:
        scale_output(correction)
    elif entry.op.tag == FA3InnerOps.ARRIVE_K.value:
        consumed_mbar_kv[read_idx].arrive()
    elif entry.op.tag == FA3InnerOps.APPLY_MASK.value:
        apply_mask(position, mask_status, kv_tile_start_row)
    elif entry.op.tag == FA3InnerOps.WAIT_MMA_N.value:
        # Framework-derived wait: wait_group[entry.op.wait_value]()
        wgmma_wait_group[entry.op.wait_value]()
```

**What this buys you:**

1. **The `wait_group[1]` is derived, not hand-chosen.** Change the
   dependency graph (add an op, change a latency) and the framework
   recomputes all wait points. No manual reasoning about "which WGMMA
   group retires first."

2. **The loop-carried p_frag dependency is explicit.** The `d=1` edge
   from SOFTMAX to WGMMA_PV makes the cross-iteration data flow visible
   in the graph, instead of hidden in the comment "copy new pfrag, used
   by `p_mul_v` on next iter."

3. **Register liveness is enforced.** The FLOW edges from WGMMA_QK to
   SOFTMAX encode that p_reg_tile can't be read until QK completes. Today
   this is a comment ("can't rw `p_reg_tile`"); with the framework, it's
   a compile-time constraint.

4. **The schedule is verifiable.** `verify_schedule()` would catch
   ordering violations — for example, if someone moved SOFTMAX before
   WAIT_QK, the verification would fail because the FLOW dependency
   isn't satisfied.

This works with the **current framework**, no extensions needed. The ops
are small enough for CSP (6 ops per iteration), the dependency model
(FLOW/ANTI with loop distances) handles the loop-carried p_frag, and the
wait derivation already computes "how many ops to wait for" — which maps
directly to `wait_group[N]`.

**Step 6 — Phase split for warp specialization.** This is the genuinely
new capability needed. AMD's ping-pong runs one interleaved program on
all warps. SM90 needs two programs from the same graph:

```mojo
# Hypothetical API extension
comptime sc = compile_schedule(FA3Schedule(...))

# Producer warp emits only load ops
comptime for i in range(sc.producer_len):
    producer_dispatch[sc.producer[i]]()

# Consumer warps emit only compute ops
comptime for i in range(sc.consumer_len):
    consumer_dispatch[sc.consumer[i]]()
```

The `ScheduleCompiler` currently emits `prologue`, `kernel`, `epilogue`.
For warp specialization, it would additionally emit `producer` and
`consumer` sublists — the same ops, partitioned by which warp group
runs them, with barrier ops injected at the synchronization points.

**Step 7 — Dispatch.** The dispatch function maps tags to NVIDIA primitives:

```mojo
@__parameter
@always_inline
def producer_dispatch[entry: ScheduleEntry]():
    comptime if entry.op.tag == FA3Ops.TMA_LOAD_K.value:
        # Wait for consumer to free stage, set expected bytes, TMA copy
        consumed_mbar_kv[stage_idx].wait(phase)
        produced_mbar_kv[stage_idx].expect_bytes(tile_bytes)
        k_tma_op.async_copy(k_tile(stage_idx), produced_mbar_kv[stage_idx], coord)
    elif entry.op.tag == FA3Ops.TMA_LOAD_V.value:
        # Same pattern for V
        consumed_mbar_kv[stage_idx].wait(phase)
        produced_mbar_kv[stage_idx].expect_bytes(tile_bytes)
        v_tma_op.async_copy(v_tile(stage_idx), produced_mbar_kv[stage_idx], coord)

@__parameter
@always_inline
def consumer_dispatch[entry: ScheduleEntry]():
    comptime if entry.op.tag == FA3Ops.WGMMA_QK.value:
        # Wait for K data, then async MMA
        produced_mbar_kv[stage_idx].wait(phase)
        warpgroup_fence(p_reg_tile)
        wgmma_0.arrive()
        wgmma_0.wgmma[...](q_smem, k_smem, p_reg_tile, ...)
        wgmma_0.commit_group()
        warpgroup_fence(p_reg_tile)
    elif entry.op.tag == FA3Ops.WGMMA_PV.value:
        # Wait for V data + softmax result
        produced_mbar_kv[stage_idx].wait(phase)
        warpgroup_fence(output_reg_tile)
        wgmma_1.arrive()
        wgmma_1.wgmma(p_frag, v_smem, output_reg_tile)
        wgmma_1.commit_group()
        warpgroup_fence(output_reg_tile)
    elif entry.op.tag == FA3Ops.SOFTMAX.value:
        wgmma_0.wait_group[0]()  # wait for Q×K to complete
        online_softmax(p_reg_tile)
```

Notice how the `warpgroup_fence` → `arrive` → `wgmma` → `commit_group`
ceremony is entirely in the dispatch function, not in the framework. The
framework decides *what* op runs *when* and *where* to synchronize. The
dispatch function knows *how* to issue each op on the actual hardware.
This is the same separation as the AMD kernels — the framework is
target-independent, the dispatch is target-specific.

### What the framework provides today vs what's needed

**Works today, no changes needed:**

- `OpDesc` / `DepEdge` / `LoopBody` — the dependency graph representation
  is target-independent. It models the FA3 pipeline correctly.
- `TargetCostModel` — latency and resource assignment per op tag. Just
  needs NVIDIA-specific values.
- `greedy_schedule()` / `optimal_schedule()` — the scheduling algorithms
  operate on the abstract graph. They find the optimal interleaving of
  TMA loads and WGMMA ops without knowing they're on NVIDIA hardware.
- `verify_schedule()` — structural checks (op count, phase partition,
  stage consistency) apply to any target.

**Needs extension:**

- **`Phase` enum**: add `PRODUCER` / `CONSUMER` variants so the compiler
  can partition ops by warp group. The prologue/epilogue derivation would
  generate per-warp-group programs.
- **Wait emission strategy**: a target-specific hook that converts abstract
  "wait for N pending ops" into either `s_waitcnt` (AMD) or
  `mbarrier.wait()` + `wait_group_sync[N]()` (NVIDIA). The analysis is
  shared; the emission differs.
- **Barrier op injection**: the framework needs to insert
  `consumed_mbar.arrive()` ops in the consumer program (to signal "I'm
  done with this stage") and `produced_mbar.arrive()` / `expect_bytes()`
  ops in the producer program. These are structural — derivable from
  the stage metadata on each op.

**Explicitly NOT needed:**

- The `ProducerConsumerPipeline` runtime abstraction already exists in
  `structured_kernels/pipeline.mojo`. The framework doesn't replace it —
  it decides which ops go in each stage at compile time, while
  `ProducerConsumerPipeline` manages the barriers at runtime.
- No changes to the scheduling algorithms. The CSP solver works on
  abstract dependency graphs — it doesn't care whether the underlying
  hardware uses counters or barriers.

## PipelineConfig reference

Key fields that every schedule must provide:

| Field            | Type   | Description                                        |
|------------------|--------|----------------------------------------------------|
| `depth`          | `Int`  | Buffer depth. 1 = single-buffer, 2 = double-buffer |
| `prefetch`       | `Int`  | How many iterations to prefetch ahead              |
| `drain_passes`   | `Int`  | Number of epilogue drain iterations                |
| `prologue_fill`  | `Int`  | Number of prologue fill iterations                 |
| `m_mmas`         | `Int`  | MMA grid rows (M-dimension tiles)                  |
| `n_mmas`         | `Int`  | MMA grid columns (N-dimension tiles)               |
| `num_partitions` | `Int`  | 1 = single-sided, 2 = ping-pong (two LDS stages)   |
| `mma_serial`     | `Bool` | True if MMA unit is serial (capacity 1)            |
| `mma_latency`    | `Int`  | Cycles per MMA (for scheduling heuristics)         |
| `vm_per_load_a`  | `Int`  | `vmcnt` ticks consumed per A-channel global load   |
| `vm_per_load_b`  | `Int`  | `vmcnt` ticks consumed per B-channel global load   |

Single-buffer pipelines set `depth=1`, `num_partitions=1`, and rely on barriers
to gate read/write phases. Double-buffer pipelines set `depth=2`,
`num_partitions=2`, and interleave operations across two LDS stages.

## ScheduleConfig reference

Tuning knobs that control the scheduling and wait-derivation algorithms:

| Field             | Type                 | Default  | Description                               |
|-------------------|----------------------|----------|-------------------------------------------|
| `scheduling`      | `SchedulingStrategy` | `MANUAL` | `MANUAL`, `GREEDY`, or `CSP`              |
| `auto_waits`      | `Bool`               | `False`  | Auto-derive `vmcnt`/`lgkmcnt` from blocks |
| `lgkm_per_load_a` | `Int`                | `0`      | lgkmcnt ticks per A-channel LDS load      |
| `lgkm_per_load_b` | `Int`                | `0`      | lgkmcnt ticks per B-channel LDS load      |

When `auto_waits=True`, the framework computes the minimum wait counts
per block from the block structure and hardware config. When `False`, you
set them manually via the config fields.

## Debugging a schedule

### Inspecting the compiled schedule

The `ScheduleCompiler` holds all intermediate results:

```mojo
comptime sc = compile_schedule(MySchedule(...))

# Inspect phase sizes
print("prologue:", len(sc.prologue), "entries")
print("kernel:", len(sc.kernel), "entries")
print("epilogue:", len(sc.epilogue), "entries")

# Inspect individual entries
comptime for i in range(len(sc.kernel)):
    comptime entry = sc.kernel[i]
    print("  tag=", entry.op.tag, "stage=", entry.op.stage,
          "subtile=", entry.op.subtile, "prefetch=", entry.is_prefetch)
```

### Dumping the program

For double-buffer schedules, `dump_program_blocks()` prints the MMA block
layout:

```mojo
from pipeline.program_builder import dump_program_blocks

comptime program = build_kernel_program(body, config, sched_config)
dump_program_blocks(program, config)
```

Output shows which ops land in which block and their stage assignments.

### Common issues

**"Cross-block LDS race" compile error**: a global load in block N writes
to the same LDS stage that block N+1 reads via fragment load. Under warp
stagger, the async write can corrupt the read. Fix: reduce `max_globals`
in `BlockSizing` or increase `num_k_mmas` for more MMA latency between
write and read.

**"Not all ops scheduled" assertion**: the dependency graph has a cycle or
an unsatisfiable constraint. Check that your edges don't form a loop.

**Performance regression after changing schedule**: the `vmcnt`/`lgkmcnt`
wait counts may be wrong. Set `auto_waits=True` in `ScheduleConfig` to
let the framework derive them, or manually inspect the derived waits via
`dump_program_blocks()`.

## Adding a new kernel schedule

Step-by-step:

1. **Define op tags**: create a struct conforming to `ScheduleOps` with your
   kernel's data operations (tags 0-127).

2. **Write op factories** (optional): convenience functions that construct
   `OpDesc` values for each op tag with the right resource, latency, and
   role. Or use `PipelineBody` + `annotate_ops()` to separate declaration
   from costs.

3. **Write the logical body**: declare what ops exist per iteration, using
   `PipelineBody` or direct `OpDesc` construction. Focus on buffer metadata
   (stage, subtile, channel, k_offset) — not hardware costs.

4. **Define a cost model**: create a `TargetCostModel` that maps each tag
   to (resource, latency, role). Or use an existing target profile.

5. **Implement `PipelineSchedule`**: at minimum, `config()` and
   `build_body()`. Override `transform_kernel()` if you need target-specific
   post-processing (like AMD schedule hints).

6. **Compile and consume**: `compile_schedule()` produces a
   `ScheduleCompiler` with prologue/kernel/epilogue as `List[ScheduleEntry]`.
   Write a dispatch function that maps tags to kernel primitives.

7. **Test**: the framework's `verify_schedule()` runs automatically during
   compilation. Add unit tests for your schedule using
   `test_pipeline_ldg.mojo` as a template.
