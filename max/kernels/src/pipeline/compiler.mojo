# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Pipeline schedule compiler: PipelineSchedule trait, ScheduleCompiler struct,
and compile_schedule function.
"""

from std.collections import List

from .config import (
    BlockSizing,
    FragOrder,
    LoopCarriedSpec,
    PipelineConfig,
    ScheduleConfig,
    WarpStaggerRule,
)
from .phase_derivation import (
    default_epilogue,
    default_kernel,
    default_prologue,
    default_warp_stagger,
    default_warp_stagger_double_buffer,
    derive_edges_from_ops,
    derive_epilogue_from_program,
    derive_prologue_from_program,
)
from .pipeline_dsl import ScheduleEntry
from .program import PipelineProgram
from .program_builder import (
    build_kernel_program,
    default_kernel_deps_double_buffer,
)
from .types import DepEdge, OpDesc, Phase

# =============================================================================
# Schedule Compiler (List-based, no magic numbers)
# =============================================================================


trait PipelineSchedule:
    """Pipeline schedule definition.

    2 required methods define the kernel-specific logic:
      - config(): pipeline structure (depth, MMA grid, etc.)
      - build_body(): the pipelined loop body

    4 optional methods have defaults:
      - derive_edges(): dependency edges (default: inferred from ops + config)
      - schedule_config(): tuning knobs (default: ScheduleConfig())
      - transform_kernel(): post-process kernel entries (default: identity)
      - bootstrap_frags(): post-prologue frag-loads (default: empty)

    The framework owns all phase derivation (prologue, kernel, epilogue,
    kernel deps). For single-buffer (depth<2): default functions + optional
    transform_kernel hook. For double-buffer (depth>=2): builds a
    PipelineProgram and derives all phases from its block structure.
    """

    # --- Required (kernel-specific) ---

    def config(self) -> PipelineConfig:
        """Pipeline configuration (depth, MMA grid, etc.)."""
        ...

    def build_body(self) -> List[OpDesc]:
        """Construct the pipelined loop body (raw data ops)."""
        ...

    # --- Defaults (framework-derived, override if needed) ---

    def derive_edges(self, body: List[OpDesc]) -> List[DepEdge]:
        """Derive LDG dependency edges. Default: inferred from op types + config.
        """
        return derive_edges_from_ops(body, self.config())

    def schedule_config(self) -> ScheduleConfig:
        """Tuning knobs for scheduling and wait derivation."""
        return ScheduleConfig()

    def transform_kernel(
        self, ker: List[ScheduleEntry], body: List[OpDesc]
    ) -> List[ScheduleEntry]:
        """Post-process kernel entries (e.g., append AMD schedule hints).

        Default: identity (return entries unchanged). Override for
        kernel-specific transformations that don't fit in the framework.
        """
        return ker.copy()

    def build_explicit_blocks(
        self,
        body: List[OpDesc],
        program: PipelineProgram,
    ) -> List[List[OpDesc]]:
        """Returns optional per-block explicit op lists, bypassing
        `MMABlockSpec.expand`'s flag-driven template.

        Return one `List[OpDesc]` per block (in block order). Empty
        entries fall back to the template; non-empty entries are
        emitted verbatim by `PipelineProgram.expand_to_list`.

        Called after the framework has populated `program.blocks`
        (frag/load/mma grouping + wait derivation) so the schedule
        can read out the analyzed block structure when constructing
        its explicit op lists.

        Use when a schedule's emission shape can't be expressed cleanly
        via the existing flag set (`global_before_frag`,
        `barrier_before_pre_ops`, `wrap_waits_with_sched_barrier`,
        etc.). The schedule constructs the exact op sequence it wants;
        the framework consumes it without templated reordering.

        Default: empty list (= every block uses the flag-driven
        template, current behaviour).

        Args:
            body: Loop body op list, as produced by `build_body`.
            program: The framework-built `PipelineProgram` whose blocks
                have already been populated with frag/load/mma analysis.

        Returns:
            One `List[OpDesc]` per program block, or an empty list to
            fall back entirely to the template path.
        """
        return List[List[OpDesc]]()

    def bootstrap_frags(self) -> List[OpDesc]:
        """Returns optional fragment loads to issue at the prologue tail.

        Each bootstrap frag is emitted by the framework as
        `wait_vm(N) + barrier + frag` where `N` partial-drains the
        vmcnt down to leave exactly the prefetch this frag depends on
        completed (and the rest in flight). The i-th bootstrap frag
        targets the i-th prefetch in prologue order; per-frag drain
        values are derived from cumulative prefetch `vm_cost`.

        Use for kernels whose first main-loop iter expects same-stage
        leading-quadrant frags pre-loaded: e.g. cross-stage rotation
        patterns where the body's sub=0 frags read the *cross* stage,
        so the same-stage versions need explicit bootstrap.

        Only fires when `ScheduleConfig.partial_prologue_drain=True`.

        Default: empty (no bootstrap).

        Returns:
            Fragment-load ops emitted at the prologue tail, in prologue
            order.
        """
        return List[OpDesc]()


struct ScheduleCompiler(Movable):
    """Generic pipeline schedule compiler.

    Orchestrates schedule construction by calling PipelineSchedule trait
    methods in order. All kernel-specific logic lives in trait implementations.
    All intermediate state is List; sizes are discovered, not prescribed.

    Usage:

    ```mojo
    var sc = ScheduleCompiler()
    sc.compile(SingleBufferSchedule[T](hints))
    # Read phases via comptime for over sc.prologue, sc.kernel, sc.epilogue
    ```
    """

    var config: PipelineConfig
    var body: List[OpDesc]
    var edges: List[DepEdge]
    var prologue: List[ScheduleEntry]
    var kernel: List[ScheduleEntry]
    var epilogue: List[ScheduleEntry]
    var kernel_deps: List[DepEdge]
    var warp_stagger_index: Int
    var lgkm_per_load_a: Int
    var lgkm_per_load_b: Int

    def __init__(out self):
        # Sentinel config — overwritten by compile().
        self.config = PipelineConfig(
            depth=0,
            prefetch=0,
            drain_passes=0,
            prologue_fill=0,
            loop_carried=LoopCarriedSpec.none(),
            block_sizing=BlockSizing.uniform(0),
            frag_order=FragOrder(b_before_a=False),
            m_mmas=0,
            n_mmas=0,
            num_partitions=0,
            mma_serial=False,
            mma_latency=0,
            vm_per_load_a=0,
            vm_per_load_b=0,
            ch0_match_field=0,
            ch1_match_field=0,
            warp_stagger=WarpStaggerRule.none(),
        )
        self.body = List[OpDesc]()
        self.edges = List[DepEdge]()
        self.prologue = List[ScheduleEntry]()
        self.kernel = List[ScheduleEntry]()
        self.epilogue = List[ScheduleEntry]()
        self.kernel_deps = List[DepEdge]()
        self.warp_stagger_index = 0
        self.lgkm_per_load_a = 0
        self.lgkm_per_load_b = 0

    # --- Trait-based compilation ---

    def compile[S: PipelineSchedule](mut self, schedule: S):
        """Full pipeline compilation from a schedule definition.

        The framework owns all phase derivation:
        - Single-buffer (depth<2): default functions + transform_kernel hook
        - Double-buffer (depth>=2): PipelineProgram → derive phases from blocks
        """
        self.config = schedule.config()
        self.body = schedule.build_body()
        self.edges = schedule.derive_edges(self.body)

        if self.config.depth >= 2:
            # Double-buffer: build program once, derive all phases from it.
            var sched = schedule.schedule_config()
            var program = build_kernel_program(
                self.body, self.config, sched, self.edges
            )
            # Schedule may override per-block emission via `build_explicit_blocks`.
            # Empty entries fall back to the flag-driven template; non-empty
            # entries are emitted verbatim by `PipelineProgram.expand_to_list`.
            var explicit = schedule.build_explicit_blocks(self.body, program)
            if len(explicit) > 0:
                debug_assert(
                    len(explicit) == len(program.blocks),
                    (
                        "build_explicit_blocks must return one entry per"
                        " program block"
                    ),
                )
                program.explicit_blocks = explicit^
            self.prologue = derive_prologue_from_program(
                program, self.config, sched, schedule.bootstrap_frags()
            )
            self.kernel = program.expand_to_list(Phase.KERNEL)
            self.epilogue = derive_epilogue_from_program(program, self.config)
            self.kernel_deps = default_kernel_deps_double_buffer(
                self.kernel, self.config
            )
        else:
            # Single-buffer: defaults + optional kernel transform.
            self.prologue = default_prologue(self.body, self.config)
            self.kernel = schedule.transform_kernel(
                default_kernel(self.body), self.body
            )
            self.epilogue = default_epilogue(self.body, self.config)
            self.kernel_deps = self.edges.copy()

        # Warp stagger from config rules.
        var ws = self.config.warp_stagger
        if not ws.enabled:
            self.warp_stagger_index = default_warp_stagger(len(self.prologue))
        elif ws.compute_from_body:
            self.warp_stagger_index = default_warp_stagger_double_buffer(
                self.body
            )
        else:
            self.warp_stagger_index = ws.fixed_amount


def compile_schedule[
    S: PipelineSchedule,
](schedule: S) -> ScheduleCompiler:
    """Compile a PipelineSchedule into a ScheduleCompiler.

    This is the compile-time entry point. The returned compiler holds
    all phases as Lists (prologue, kernel, epilogue, kernel_deps).
    Use as a comptime value; the kernel reads entries via comptime for.

    Parameters:
        S: The PipelineSchedule implementation type.
    """
    var sc = ScheduleCompiler()
    sc.compile(schedule)
    return sc^
