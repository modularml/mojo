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

"""Figure 21.5: DCS Scatter Kernel (GPU implementation).

This kernel uses the scatter pattern where each thread handles one atom
and iterates over all grid points, using atomic operations to accumulate
energy contributions.

Note: Scatter pattern has poor parallelism for few atoms and requires
atomic operations which can be a performance bottleneck.
"""

from std.math import sqrt
from std.atomic import Atomic
from max.gpu import block_idx, thread_idx, block_dim
from max.gpu.host import DeviceContext

from dcs_utils import GridDim, init_atoms, verify_grid


def cenergy_scatter_kernel(
    energygrid: UnsafePointer[Float32, MutAnyOrigin],
    atoms: UnsafePointer[Float32, ImmutAnyOrigin],
    grid_x_dev: Int32,
    grid_y_dev: Int32,
    gridspacing: Float32,
    z: Float32,
    numatoms_dev: Int32,
):
    """Scatter kernel: each thread processes one atom.

    Each thread computes the energy contribution of one atom to all grid
    points and uses atomic add to accumulate the results.

    Args:
        energygrid: Output energy grid.
        atoms: Atom array (4 floats per atom: x, y, z, charge).
        grid_x_dev: Grid X dimension.
        grid_y_dev: Grid Y dimension.
        gridspacing: Grid spacing.
        z: Z-coordinate of the grid slice.
        numatoms_dev: Number of atoms.
    """
    # Int is not device-passable; widen the fixed-width args.
    var grid_x = Int(grid_x_dev)
    var grid_y = Int(grid_y_dev)
    var numatoms = Int(numatoms_dev)
    # Each thread handles one atom
    var n = (block_idx.x * block_dim.x + thread_idx.x) * 4

    # Bounds check for atom array
    if n < numatoms * 4:
        var dz = z - atoms[n + 2]
        var dz2 = dz * dz

        # Calculate slice offset in energy grid
        var k = Int(z / gridspacing + 0.5)
        var grid_slice_offset = grid_x * grid_y * k

        var charge = atoms[n + 3]

        for j in range(grid_y):
            var y = gridspacing * Float32(j)
            var dy = y - atoms[n + 1]
            var dy2 = dy * dy

            var grid_row_offset = grid_slice_offset + grid_x * j

            for i in range(grid_x):
                var x = gridspacing * Float32(i)
                var dx = x - atoms[n]

                # Scatter add with atomic operation
                var energy_contrib = charge / sqrt(dx * dx + dy2 + dz2)
                _ = Atomic.fetch_add(
                    energygrid + grid_row_offset + i, energy_contrib
                )


def cenergy_cpu_reference(
    energygrid: UnsafePointer[mut=True, Float32, _],
    grid: GridDim,
    gridspacing: Float32,
    z: Float32,
    atoms: UnsafePointer[mut=False, Float32, _],
    numatoms: Int,
):
    """CPU reference implementation for verification."""
    var atomarrdim = numatoms * 4
    var k = Int(z / gridspacing + 0.5)

    for n in range(0, atomarrdim, 4):
        var dz = z - atoms[n + 2]
        var dz2 = dz * dz
        var charge = atoms[n + 3]

        for j in range(grid.y):
            var y = gridspacing * Float32(j)
            var dy = y - atoms[n + 1]
            var dy2 = dy * dy

            for i in range(grid.x):
                var x = gridspacing * Float32(i)
                var dx = x - atoms[n]

                energygrid[
                    grid.x * grid.y * k + grid.x * j + i
                ] += charge / sqrt(dx * dx + dy2 + dz2)


def main() raises:
    print("Figure 21.5: DCS Scatter Kernel")

    var ctx = DeviceContext()

    # Parameters
    var numatoms = 256
    var vol_dim = GridDim(64, 64, 1)
    var gridspacing: Float32 = 0.5
    var z_coord: Float32 = 0.0

    var atoms_size = numatoms * 4
    var grid_size = vol_dim.x * vol_dim.y * vol_dim.z

    # Allocate host memory
    var h_atoms = alloc[Float32](atoms_size)
    var h_energygrid_cpu = alloc[Float32](grid_size)
    var h_energygrid_gpu = alloc[Float32](grid_size)

    # Initialize
    init_atoms(h_atoms, numatoms, gridspacing, vol_dim.x, vol_dim.y, vol_dim.z)
    for i in range(grid_size):
        h_energygrid_cpu[i] = 0.0
        h_energygrid_gpu[i] = 0.0

    # Allocate device memory
    var d_atoms = ctx.enqueue_create_buffer[.float32](atoms_size)
    var d_energygrid = ctx.enqueue_create_buffer[.float32](grid_size)

    # Copy atoms to device
    ctx.enqueue_copy(d_atoms, h_atoms)

    # Initialize device grid to zero
    var h_zeros = alloc[Float32](grid_size)
    for i in range(grid_size):
        h_zeros[i] = 0.0
    ctx.enqueue_copy(d_energygrid, h_zeros)
    h_zeros.free()

    # Launch config: one thread per atom
    var block_size = 256
    var num_blocks = (numatoms + block_size - 1) // block_size

    ctx.enqueue_function[cenergy_scatter_kernel](
        d_energygrid,
        d_atoms,
        Int32(vol_dim.x),
        Int32(vol_dim.y),
        gridspacing,
        z_coord,
        Int32(numatoms),
        grid_dim=(num_blocks, 1, 1),
        block_dim=(block_size, 1, 1),
    )
    ctx.synchronize()

    # Copy results back
    ctx.enqueue_copy(h_energygrid_gpu, d_energygrid)
    ctx.synchronize()

    # CPU verification
    cenergy_cpu_reference(
        h_energygrid_cpu, vol_dim, gridspacing, z_coord, h_atoms, numatoms
    )

    if verify_grid(h_energygrid_cpu, h_energygrid_gpu, vol_dim.x * vol_dim.y):
        print("Figure 21.5 (Scatter) Passed!")
    else:
        print("Figure 21.5 (Scatter) Failed!")

    # Cleanup
    h_atoms.free()
    h_energygrid_cpu.free()
    h_energygrid_gpu.free()
