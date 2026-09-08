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
"""Grid Dependent Control primitives for NVIDIA Hopper (SM90+) GPUs.

This module provides low-level primitives for managing grid dependencies on NVIDIA
Hopper architecture and newer GPUs. It enables efficient orchestration of multi-grid
workloads by allowing grids to launch dependent grids and synchronize with them.

The module includes functions that map directly to CUDA grid dependency control
instructions, providing fine-grained control over grid execution order:

- `launch_dependent_grids()`: Triggers execution of grids that depend on the
  current grid
- `wait_on_dependent_grids()`: Blocks until all dependent grids complete execution

These primitives are essential for implementing complex GPU execution pipelines where
multiple kernels need to execute in a specific order with minimal overhead. They
eliminate the need for host-side synchronization when orchestrating dependent GPU work.
"""
from max.gpu.host.launch_attribute import (
    LaunchAttribute,
    LaunchAttributeID,
    LaunchAttributeValue,
)
from std.sys import has_nvidia_gpu_accelerator
from std.sys.info import _accelerator_arch

from ..host.info import H100, GPUInfo

comptime _SUPPORT_PDL_LAUNCH = _support_pdl_launch()


@doc_hidden
@always_inline("nodebug")
def _support_pdl_launch() -> Bool:
    """Determines if programmatic dependency launch (PDL) is supported.

    Checks if the current GPU supports PDL (Hopper SM90+ architecture).
    Returns False for unsupported GPUs

    Returns:
        True if PDL is supported and enabled, False otherwise.
    """

    comptime if (
        has_nvidia_gpu_accelerator()
        and GPUInfo.from_name[_accelerator_arch()]().compute >= H100.compute
    ):
        return True
    else:
        return False


@doc_hidden
@always_inline("nodebug")
def pdl_launch_attributes(
    pdl_level: PDLLevel = PDLLevel(),
) -> List[LaunchAttribute]:
    """Returns launch attributes for programmatic dependency launch (PDL).

    This function configures launch attributes to enable programmatic stream
    serialization on supported GPUs. When PDL is enabled, it returns a list
    containing a single launch attribute that enables grid dependency control.

    Returns:
        A list of launch attributes. Contains the PDL attribute if enabled,
        otherwise returns an empty list.

    Note:
        - Only supported on NVIDIA SM90+ (Hopper architecture and newer) GPUs.
        - When disabled, returns an empty list for compatibility with older GPUs.
    """

    if _SUPPORT_PDL_LAUNCH and pdl_level != PDLLevel.OFF:
        return [
            LaunchAttribute(
                LaunchAttributeID.PROGRAMMATIC_STREAM_SERIALIZATION,
                LaunchAttributeValue(True),
            )
        ]
    else:
        return List[LaunchAttribute]()


@always_inline("nodebug")
def launch_dependent_grids():
    """Launches dependent grids that were previously configured to depend on the
    current grid.

    This function triggers the execution of dependent grids that have been configured
    with a dependency on the current grid. It maps directly to the CUDA grid
    dependency control instruction for launching dependent grids.

    Note:
        - Only supported on NVIDIA SM90+ (Hopper architecture and newer) GPUs.
        - Must be called by all threads in a thread block to avoid undefined behavior.
        - Typically used in multi-grid pipeline scenarios where one grid's completion
          should trigger the execution of other grids.
    """

    comptime if _SUPPORT_PDL_LAUNCH:
        comptime kind_attr = __mlir_attr.`#nvvm.grid_dep_action<launch_dependents>`
        __mlir_op.`nvvm.griddepcontrol`[kind=kind_attr, _type=None]()


@always_inline("nodebug")
def wait_on_dependent_grids():
    """Waits for all dependent grids launched by this grid to complete execution.

    This function blocks the calling grid until all dependent grids that were launched
    by this grid have completed their execution. It provides a synchronization point
    between parent and child grids in a multi-grid dependency chain.

    Note:
        - Only supported on NVIDIA SM90+ (Hopper architecture and newer) GPUs.
        - Must be called by all threads in a thread block to avoid undefined behavior.
        - Can be used to ensure dependent grid work is complete before proceeding
          with subsequent operations in the parent grid.
    """

    comptime if _SUPPORT_PDL_LAUNCH:
        comptime kind_attr = __mlir_attr.`#nvvm.grid_dep_action<wait>`
        __mlir_op.`nvvm.griddepcontrol`[kind=kind_attr, _type=None]()


@fieldwise_init
struct PDLLevel(Defaultable, Equatable, TrivialRegisterPassable):
    """Programmatic Dependency Launch (PDL) level."""

    var _level: Int

    comptime OFF = PDLLevel(0)
    """PDL disabled."""

    comptime ON = PDLLevel(1)
    """PDL enabled with default behavior."""

    comptime OVERLAP_AT_END = PDLLevel.ON
    """PDL overlap at end of kernel."""

    comptime OVERLAP_AT_BEGINNING = PDLLevel(2)
    """PDL overlap at beginning of kernel."""

    comptime NO_WAIT_OVERLAP_AT_END = PDLLevel(3)
    """PDL no-wait overlap at end of kernel."""

    @always_inline
    def __init__(out self):
        """Initialize the PDL level to OFF."""
        self = PDLLevel.OFF

    @always_inline
    def __eq__(self, other: Int) -> Bool:
        """Check if the PDL level is equal to another PDL level.

        Args:
            other: The other PDL level to compare against.

        Returns:
            True if the PDL level is equal to the other PDL level, False otherwise.
        """
        return self._level == other

    @always_inline
    def __gt__(self, other: PDLLevel) -> Bool:
        """Check if the PDL level is greater than another PDL level.

        Args:
            other: The other PDL level to compare against.

        Returns:
            True if the PDL level is greater than the other PDL level, False otherwise.
        """
        return self._level > other._level

    @always_inline
    def __ge__(self, other: PDLLevel) -> Bool:
        """Check if the PDL level is greater than or equal to another PDL level.

        Args:
            other: The other PDL level to compare against.

        Returns:
            True if the PDL level is greater or equal to the other PDL level,
            False otherwise.
        """
        return self._level >= other._level


struct PDL[overlap_at_beginning: Bool = False](Defaultable):
    """Programmatic Dependency Launch (PDL) control structure.

    This struct provides a way to manage programmatic stream serialization on
    NVIDIA GPUs. It waits on the predecessor grids on entry and releases the
    dependent grids on exit.

    `overlap_at_beginning` (`PDLLevel.OVERLAP_AT_BEGINNING`) releases the
    dependents on *entry* instead, so they turn resident while this kernel is
    still running and overlap it with their predecessor-independent work (weight
    loads, descriptor setup). Releasing early cannot expose unwritten output: a
    dependent's own `wait_on_dependent_grids` blocks until the predecessor grids
    have completed and their stores are visible, so the release only decides
    when the dependent is scheduled. It is unsafe only for a dependent that
    writes memory this grid still reads, before its own wait.

    Parameters:
        overlap_at_beginning: Release the dependent grids on entry rather than
            on exit.

    Note:
        - Only supported on NVIDIA SM90+ (Hopper architecture and newer) GPUs.
    """

    @always_inline
    def __init__(out self):
        """Initialize the PDL control structure."""
        pass

    @always_inline
    def __enter__(self):
        """Wait for the predecessor grids to complete."""
        wait_on_dependent_grids()
        comptime if Self.overlap_at_beginning:
            launch_dependent_grids()

    @always_inline
    def __exit__(self):
        """Release the grids that depend on this one."""
        comptime if not Self.overlap_at_beginning:
            launch_dependent_grids()
