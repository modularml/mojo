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

"""Figure 21.8: DCS Kernel with Thread Coarsening (GPU implementation).

This kernel uses thread coarsening where each thread computes multiple
consecutive grid points (COARSEN_FACTOR points). This reduces the number
of threads needed and can improve efficiency by reusing atom data across
multiple grid point calculations.
"""

from std.math import sqrt
from max.gpu import block_idx, thread_idx, block_dim
from max.gpu.host import DeviceContext
from std.itertools import product

from dcs_utils import GridDim, init_atoms, verify_grid


comptime COARSEN_FACTOR = 4


def cenergy_coarsening_kernel(
    energygrid: UnsafePointer[Float32, MutAnyOrigin],
    atoms: UnsafePointer[Float32, ImmutAnyOrigin],
    grid_x_dev: Int32,
    grid_y_dev: Int32,
    gridspacing: Float32,
    z: Float32,
    numatoms_dev: Int32,
):
    """Coarsening kernel: each thread computes COARSEN_FACTOR consecutive points.

    The coarsening is applied along the X dimension. Each thread handles
    grid points at i, i+1, i+2, i+3 (for COARSEN_FACTOR=4).

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
    # Multiply threadIdx by COARSEN_FACTOR to avoid overlap
    var i = (block_idx.x * block_dim.x + thread_idx.x) * COARSEN_FACTOR
    var j = block_idx.y * block_dim.y + thread_idx.y

    if i < grid_x and j < grid_y:
        var atomarrdim = numatoms * 4
        var k = Int(z / gridspacing + 0.5)

        var y = gridspacing * Float32(j)
        var x = gridspacing * Float32(i)

        var energy0: Float32 = 0.0
        var energy1: Float32 = 0.0
        var energy2: Float32 = 0.0
        var energy3: Float32 = 0.0

        for n in range(0, atomarrdim, 4):
            var dx0 = x - atoms[n]
            # Sequential points handled by this thread
            var dx1 = dx0 + gridspacing
            var dx2 = dx0 + 2 * gridspacing
            var dx3 = dx0 + 3 * gridspacing

            var dy = y - atoms[n + 1]
            var dz = z - atoms[n + 2]
            var dysqdzsq = dy * dy + dz * dz
            var charge = atoms[n + 3]

            energy0 += charge / sqrt(dx0 * dx0 + dysqdzsq)
            energy1 += charge / sqrt(dx1 * dx1 + dysqdzsq)
            energy2 += charge / sqrt(dx2 * dx2 + dysqdzsq)
            energy3 += charge / sqrt(dx3 * dx3 + dysqdzsq)

        var base_idx = grid_x * grid_y * k + grid_x * j + i

        # Boundary checks for coarsened points
        if i < grid_x:
            energygrid[base_idx] += energy0
        if i + 1 < grid_x:
            energygrid[base_idx + 1] += energy1
        if i + 2 < grid_x:
            energygrid[base_idx + 2] += energy2
        if i + 3 < grid_x:
            energygrid[base_idx + 3] += energy3


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
    print("Figure 21.8: DCS Kernel with Thread Coarsening")

    var ctx = DeviceContext()

    var numatoms = 256
    var vol_dim = GridDim(128, 128, 1)  # Larger grid to justify coarsening
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

    # Grid X dimension is reduced by COARSEN_FACTOR
    comptime BLOCK_DIM_X = 16
    comptime BLOCK_DIM_Y = 16
    var grid_dim_x = (vol_dim.x + BLOCK_DIM_X * COARSEN_FACTOR - 1) // (
        BLOCK_DIM_X * COARSEN_FACTOR
    )
    var grid_dim_y = (vol_dim.y + BLOCK_DIM_Y - 1) // BLOCK_DIM_Y

    ctx.enqueue_function[cenergy_coarsening_kernel](
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
        print("Figure 21.8 (Coarsening) Passed!")
    else:
        print("Figure 21.8 (Coarsening) Failed!")

    h_atoms.free()
    h_energygrid_cpu.free()
    h_energygrid_gpu.free()
