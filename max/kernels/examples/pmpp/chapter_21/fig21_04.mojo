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

"""Figure 21.4: Optimized DCS C code (CPU implementation).

This version reorders the loops to have atoms in the outermost loop,
which reduces redundant computations of dz and dz^2. The z-related
calculations are hoisted out of the inner loops.
"""

from std.math import sqrt

from dcs_utils import GridDim, init_atoms, compute_total_energy


def cenergy_cpu_optimized(
    energygrid: UnsafePointer[mut=True, Float32, _],
    grid: GridDim,
    gridspacing: Float32,
    z: Float32,
    atoms: UnsafePointer[mut=False, Float32, _],
    numatoms: Int,
):
    """Optimized CPU implementation of DCS energy calculation.

    This function reorders the loops: atoms -> j (rows) -> i (cols).
    This allows dz and dz^2 to be computed once per atom, and dy and dy^2
    once per row.

    Args:
        energygrid: Output energy grid.
        grid: Grid dimensions.
        gridspacing: Grid spacing.
        z: Z-coordinate of the grid slice.
        atoms: Atom array (4 floats per atom: x, y, z, charge).
        numatoms: Number of atoms.
    """
    var atomarrdim = numatoms * 4
    var k = Int(z / gridspacing + 0.5)
    var grid_slice_offset = grid.x * grid.y * k

    # Atom loop is outermost for optimization
    for n in range(0, atomarrdim, 4):
        var dz = z - atoms[n + 2]
        var dz2 = dz * dz
        var charge = atoms[n + 3]

        for j in range(grid.y):
            var y = gridspacing * Float32(j)
            var dy = y - atoms[n + 1]
            var dy2 = dy * dy
            var grid_row_offset = grid_slice_offset + grid.x * j

            for i in range(grid.x):
                var x = gridspacing * Float32(i)
                var dx = x - atoms[n]

                # Accumulate energy contribution
                energygrid[grid_row_offset + i] += charge / sqrt(
                    dx * dx + dy2 + dz2
                )


def main() raises:
    print("Figure 21.4: CPU Optimized DCS")

    var grid = GridDim(64, 64, 1)
    var gridspacing: Float32 = 0.5
    var z_coord: Float32 = 0.0
    var numatoms = 256

    var atoms_size = numatoms * 4
    var grid_size = grid.x * grid.y

    var h_atoms = alloc[Float32](atoms_size)
    var h_energygrid = alloc[Float32](grid_size)

    init_atoms(h_atoms, numatoms, gridspacing, grid.x, grid.y, 1)
    for i in range(grid_size):
        h_energygrid[i] = 0.0

    cenergy_cpu_optimized(
        h_energygrid, grid, gridspacing, z_coord, h_atoms, numatoms
    )

    print("Figure 21.4 CPU Optimized Execution Completed.")

    var total_energy = compute_total_energy(h_energygrid, grid_size)
    print("Total Energy:", total_energy)

    h_atoms.free()
    h_energygrid.free()
