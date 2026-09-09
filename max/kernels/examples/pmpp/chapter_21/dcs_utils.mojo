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

"""Direct Coulomb Summation (DCS) utilities for Chapter 21.

This module provides common utilities for DCS implementations including:
- Grid dimension structure
- Atom initialization
- Verification functions
"""

from std.random import random_float64, seed


comptime MAX_ATOMS = 4000  # Fits in 64KB constant memory (4000 * 4 * 4 bytes = 64KB)


struct GridDim(TrivialRegisterPassable):
    """Grid dimension structure for 3D grids."""

    var x: Int
    var y: Int
    var z: Int

    def __init__(out self, x: Int, y: Int, z: Int):
        self.x = x
        self.y = y
        self.z = z


def init_atoms(
    atoms: UnsafePointer[mut=True, Float32, _],
    num_atoms: Int,
    grid_spacing: Float32,
    grid_x: Int,
    grid_y: Int,
    grid_z: Int,
):
    """Initialize atom positions and charges.

    Atoms are placed randomly within the grid volume with random charges.

    Args:
        atoms: Pointer to atom array (4 floats per atom: x, y, z, charge).
        num_atoms: Number of atoms.
        grid_spacing: Grid spacing.
        grid_x: Grid X dimension.
        grid_y: Grid Y dimension.
        grid_z: Grid Z dimension.
    """
    seed(42)

    var max_x = grid_spacing * Float32(grid_x)
    var max_y = grid_spacing * Float32(grid_y)
    var max_z = grid_spacing * Float32(grid_z)

    for i in range(num_atoms):
        # Position within grid volume
        atoms[i * 4 + 0] = Float32(random_float64() * Float64(max_x))
        atoms[i * 4 + 1] = Float32(random_float64() * Float64(max_y))
        atoms[i * 4 + 2] = Float32(random_float64() * Float64(max_z))
        # Charge between 1.0 and 10.0
        atoms[i * 4 + 3] = Float32(1.0 + random_float64() * 9.0)


def verify_grid(
    grid_ref: UnsafePointer[mut=False, Float32, _],
    grid_test: UnsafePointer[mut=False, Float32, _],
    size: Int,
) -> Bool:
    """Verify computed energy grid against reference.

    Args:
        grid_ref: Reference energy grid.
        grid_test: Test energy grid.
        size: Number of grid points.

    Returns:
        True if grids match within tolerance, False otherwise.
    """
    var max_diff: Float32 = 0.0
    comptime epsilon: Float32 = 1e-2  # Atomic adds order can cause deviation

    for i in range(size):
        var ref_val = grid_ref[i]
        var test_val = grid_test[i]
        var diff = abs(ref_val - test_val)

        if diff > max_diff:
            max_diff = diff

        # Check for NaN or Inf
        if (
            test_val != test_val
            or test_val == Float32.MAX
            or test_val == -Float32.MAX
        ):
            print("NaN/Inf at", i)
            return False

    print("Max Diff:", max_diff)
    if max_diff > epsilon:
        print("Verification Failed!")
        return False
    return True


def compute_total_energy(
    energygrid: UnsafePointer[mut=False, Float32, _], size: Int
) -> Float64:
    """Compute total energy by summing all grid points.

    Args:
        energygrid: Energy grid.
        size: Number of grid points.

    Returns:
        Total energy.
    """
    var total: Float64 = 0.0
    for i in range(size):
        total += Float64(energygrid[i])
    return total
