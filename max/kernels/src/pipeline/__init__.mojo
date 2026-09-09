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
"""Generic compile-time software pipelining for GPU kernels: schedule
generation and verification.

This package simplifies defining a pipelined GPU kernel schedule: the
ordered list of operations in a loop (global loads, shared-memory reads,
MMA instructions, barriers, and hardware waits) for each execution
phase. For example, in a tiled matmul kernel, that ordering overlaps
fetching the next A/B tile from DRAM into LDS with multiplying the
current tile in registers. The schedule splits into a prologue (prime
buffers before overlap is possible), a kernel phase (steady overlap on
every iteration), and an epilogue (drain without issuing new loads).

You describe the operations in one iteration (loads, fragment reads,
MMA) and a per-target cost model for your GPU. At `comptime`, this
package builds a dependency graph from those declarations, schedules the
operations, derives hardware wait counts and barriers, and verifies
structural safety. You consume the result as a flat `ScheduleEntry` list
that unrolls into straight-line code with no runtime scheduling cost.

Import from submodules directly:

```mojo
from pipeline.types import OpDesc, ResourceKind, OpRole
from pipeline.config import PipelineConfig, ScheduleConfig
from pipeline.compiler import PipelineSchedule, compile_schedule
from pipeline.schedulers import optimal_schedule_with_halves
from pipeline.program_builder import verify_schedule, build_kernel_program
```
"""
