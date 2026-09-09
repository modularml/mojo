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
"""Loop Dependency Graph (LDG) types: OpNode and LoopBody."""

from std.collections import List

from .types import DepEdge, OpDesc, ResourceKind

# =============================================================================
# Operation Node (for Loop Dependency Graph)
# =============================================================================


struct OpNode(ImplicitlyCopyable, Movable):
    """An operation in the Loop Dependency Graph (LDG).

    Enriches OpDesc with resource assignment and latency, the information
    needed for modulo scheduling (GAG96). Each OpNode represents one vertex
    in the LDG.

    Latencies are approximate and used for scheduling heuristics (ASAP/ALAP
    priority), not for cycle-accurate simulation. Typical values:

    ```text
    GLOBAL_MEM (buffer_load → LDS): ~200 cycles
    LDS (ds_read → register):       ~20 cycles
    MMA_UNIT (MFMA execution):      ~16 cycles
    SCALAR (barriers, fences):      ~0 cycles (sync, not compute)
    ```
    """

    var op: OpDesc
    """The underlying `OpDesc` with kind, stage, subtile, etc."""
    var resource: ResourceKind
    """Hardware unit that executes this op (from `OpDesc.resource`)."""
    var latency: Int
    """Estimated cycles until the result is available."""

    @always_inline
    def __init__(
        out self,
        *,
        op: OpDesc,
        latency: Int = -1,
    ):
        """Create an OpNode from an OpDesc.

        If latency is -1 (default), it is auto-derived from
        op.latency. Pass an explicit value to override.
        """
        self.op = op
        self.resource = op.resource
        self.latency = op.latency if latency < 0 else latency


# =============================================================================
# Loop Dependency Graph (LDG)
# =============================================================================


struct LoopBody(Copyable, Movable):
    """Declarative specification of one loop iteration's operations and deps.

    A directed graph where vertices are OpNode values (operations with
    resource + latency) and edges are DepEdge values (producer→consumer
    with loop distance d). Engineers specify WHAT (operations +
    dependencies), and the framework derives WHEN (schedule order) and
    HOW (synchronization primitives).

    Currently implemented:
      1. ASAP/ALAP times for priority-based scheduling
      2. Structural validation (bounds, self-loop detection)
      3. Synchronization derivation (wait counts, barriers)

    Not yet implemented (future work):
      - MII = max(ResMII, RecMII) initiation interval derivation
      - C2 (data dependence) and C4 (resource contention) validation

    The graph design draws on LLVM's ScheduleDAG layering and modulo
    scheduling theory (Rau 1994, GAG96), but the current scheduler
    optimizes within-iteration makespan, not inter-iteration initiation
    interval. See DESIGN.md for scope and limitations.
    """

    var ops: List[OpNode]
    var edges: List[DepEdge]

    @always_inline
    def __init__(out self):
        self.ops = List[OpNode]()
        self.edges = List[DepEdge]()

    # =========================================================================
    # ASAP / ALAP Scheduling
    # =========================================================================

    def _propagate_times(self, mut times: List[Int], *, forward: Bool):
        """Fixed-point propagation shared by ASAP and ALAP.

        For a DAG with N nodes, at most N passes suffice (each pass relaxes
        at least one node to its final value). Only same-iteration edges
        (loop_distance == 0) are considered.

        Args:
            times: Mutable list of per-op times to propagate.
            forward: If True, propagate forward (ASAP: producer → consumer,
                candidate = times[p] + latency[p], update if greater).
                If False, propagate backward (ALAP: consumer → producer,
                candidate = times[c] - latency[p], update if less).
        """
        for _pass in range(len(self.ops)):
            var changed = False
            for e in range(len(self.edges)):
                var edge = self.edges[e]
                if edge.loop_distance >= 1:
                    continue
                var p = edge.producer_idx
                var c = edge.consumer_idx
                if forward:
                    var candidate = times[p] + self.ops[p].latency
                    if candidate > times[c]:
                        times[c] = candidate
                        changed = True
                else:
                    var candidate = times[c] - self.ops[p].latency
                    if candidate < times[p]:
                        times[p] = candidate
                        changed = True
            if not changed:
                break

    def compute_asap(self) -> List[Int]:
        """Compute ASAP (As Soon As Possible) times for each operation.

        ASAP(op) = max over all predecessors p of:

        ```text
        ASAP(p) + latency(p)   if d == 0 (same iteration)
        0                      if d >= 1 (loop-carried, skip)
        ```

        Operations with no same-iteration predecessors get ASAP = 0.
        This is computed via a fixed-point iteration (handles any DAG order).
        """
        var asap = List[Int](length=len(self.ops), fill=0)
        self._propagate_times(asap, forward=True)
        return asap^

    def compute_alap(self, makespan: Int) -> List[Int]:
        """Compute ALAP (As Late As Possible) times for each operation.

        ALAP(op) = min over all same-iteration successors s of:

        ```text
        ALAP(s) - latency(op)
        ```

        Operations with no same-iteration successors get
        ALAP = makespan - latency(op).

        This is the backward dual of compute_asap(). The difference
        ALAP(op) - ASAP(op) gives the scheduling slack (mobility).

        Note: currently used in tests only. Future schedulers may use
        ALAP-based priority to improve ordering heuristics.

        Args:
            makespan: Target schedule length to compute ALAP relative to.
                Typically the ASAP critical path length.
        """
        var alap = List(
            length=len(self.ops),
            fill_with=lambda (i: Int) -> Int: makespan - self.ops[i].latency,
        )
        self._propagate_times(alap, forward=False)
        return alap^

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self) raises:
        """Validate the loop body graph structure.

        Checks:
          - All edge indices are in bounds
          - No self-loops with d=0
          - Resource assignments are consistent with op tag
        """
        for i in range(len(self.edges)):
            var edge = self.edges[i]
            if not (
                edge.producer_idx < len(self.ops)
                and edge.consumer_idx < len(self.ops)
            ):
                raise "LoopBody: edge index out of bounds"
            if edge.loop_distance == 0:
                if edge.producer_idx == edge.consumer_idx:
                    raise "LoopBody: self-loop with d=0"
