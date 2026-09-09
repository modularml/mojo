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

"""Figure 21.6: DCS Gather Kernel (GPU implementation).

This kernel uses the gather pattern where each thread handles one grid point
and iterates over all atoms to compute the total energy at that point.

The gather pattern provides better parallelism (one thread per grid point)
and avoids atomic operations since each grid point is written by only one thread.
"""

from std.math import sqrt
from max.gpu import block_idx, thread_idx, block_dim
from max.gpu.host import DeviceContext
from std.itertools import product

from dcs_utils import GridDim, init_atoms, verify_grid


def cenergy_gather_kernel(
    energygrid: UnsafePointer[Float32, MutAnyOrigin],
    atoms: UnsafePointer[Float32, ImmutAnyOrigin],
    grid_x_dev: Int32,
    grid_y_dev: Int32,
    gridspacing: Float32,
    z: Float32,
    numatoms_dev: Int32,
):
    """Gather kernel: each thread computes one grid point.

    Each thread computes the total energy at one grid point by iterating
    over all atoms and summing their contributions.

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
    var i = block_idx.x * block_dim.x + thread_idx.x
    var j = block_idx.y * block_dim.y + thread_idx.y

    if i < grid_x and j < grid_y:
        var atomarrdim = numatoms * 4
        var k = Int(z / gridspacing + 0.5)

        var y = gridspacing * Float32(j)
        var x = gridspacing * Float32(i)
        var energy: Float32 = 0.0

        for n in range(0, atomarrdim, 4):
            var dx = x - atoms[n]
            var dy = y - atoms[n + 1]
            var dz = z - atoms[n + 2]
            energy += atoms[n + 3] / sqrt(dx * dx + dy * dy + dz * dz)

        energygrid[grid_x * grid_y * k + grid_x * j + i] += energy


def cenergy_cpu_reference(
    energygrid: UnsafePointer[mut=True, Float32, _],
    grid: GridDim,
    gridspacing: Float32,
    z: Float32,
    atoms: UnsafePointer[mut=False, Float32, _],
    numatoms: Int,
):
    """CPU reference implementation for verification."""
    var k = Int(z / gridspacing + 0.5)

    for j, i in product(range(grid.y), range(grid.x)):
        var y = gridspacing * Float32(j)
        var x = gridspacing * Float32(i)
        var energy: Float32 = 0.0

        for n in range(0, numatoms * 4, 4):
            var dx = x - atoms[n]
            var dy = y - atoms[n + 1]
            var dz = z - atoms[n + 2]
            energy += atoms[n + 3] / sqrt(dx * dx + dy * dy + dz * dz)

        energygrid[grid.x * grid.y * k + grid.x * j + i] = energy


def main() raises:
    print("Figure 21.6: DCS Gather Kernel")

    var ctx = DeviceContext()

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

    init_atoms(h_atoms, numatoms, gridspacing, vol_dim.x, vol_dim.y, vol_dim.z)
    for i in range(grid_size):
        h_energygrid_cpu[i] = 0.0
        h_energygrid_gpu[i] = 0.0

    # Allocate device memory
    var d_atoms = ctx.enqueue_create_buffer[.float32](atoms_size)
    var d_energygrid = ctx.enqueue_create_buffer[.float32](grid_size)

    ctx.enqueue_copy(d_atoms, h_atoms)

    # Initialize device grid to zero
    var h_zeros = alloc[Float32](grid_size)
    for i in range(grid_size):
        h_zeros[i] = 0.0
    ctx.enqueue_copy(d_energygrid, h_zeros)
    h_zeros.free()

    # Launch config: one thread per grid point (2D grid)
    comptime BLOCK_DIM_X = 16
    comptime BLOCK_DIM_Y = 16
    var grid_dim_x = (vol_dim.x + BLOCK_DIM_X - 1) // BLOCK_DIM_X
    var grid_dim_y = (vol_dim.y + BLOCK_DIM_Y - 1) // BLOCK_DIM_Y

    ctx.enqueue_function[cenergy_gather_kernel](
        d_energygrid,
        d_atoms,
        Int32(vol_dim.x),
        Int32(vol_dim.y),
        gridspacing,
        z_coord,
        Int32(numatoms),
        grid_dim=(grid_dim_x, grid_dim_y, 1),
        block_dim=(BLOCK_DIM_X, BLOCK_DIM_Y, 1),
    )
    ctx.synchronize()

    ctx.enqueue_copy(h_energygrid_gpu, d_energygrid)
    ctx.synchronize()

    # CPU verification
    cenergy_cpu_reference(
        h_energygrid_cpu, vol_dim, gridspacing, z_coord, h_atoms, numatoms
    )

    if verify_grid(h_energygrid_cpu, h_energygrid_gpu, vol_dim.x * vol_dim.y):
        print("Figure 21.6 (Gather) Passed!")
    else:
        print("Figure 21.6 (Gather) Failed!")

    h_atoms.free()
    h_energygrid_cpu.free()
    h_energygrid_gpu.free()
