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
"""Tests that TileTensor load/store generate correct GPU assembly instructions.

Verifies PTX instruction selection for global memory, shared memory, invariant,
and vectorized load/store operations when compiled for NVIDIA GPUs.
"""

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from std.testing import assert_true, TestSuite

from layout import (
    ComptimeInt,
    CoordLike,
    Idx,
    RowMajorLayout,
    TileTensor,
    coord,
    row_major,
)

# Target SM80 (Ampere) explicitly so the test cross-compiles to NVIDIA PTX
# regardless of the host GPU.
comptime _SM80 = get_gpu_target["sm_80"]()

comptime _4x4 = RowMajorLayout[ComptimeInt[4], ComptimeInt[4]]


# ===-----------------------------------------------------------------------===#
# Kernel functions compiled to PTX for assembly verification.
# ===-----------------------------------------------------------------------===#


def global_load_store_kernel(
    tensor: TileTensor[
        mut=True, .float32, _4x4, MutAnyOrigin, address_space=.GLOBAL
    ],
):
    """Load and store through a TileTensor backed by global memory."""
    var val = tensor.load(coord[0, 0])
    tensor.store(coord[1, 0], val)


def shared_load_kernel(
    src: TileTensor[.float32, _4x4, MutAnyOrigin, address_space=.SHARED],
    dst: TileTensor[
        mut=True, .float32, _4x4, MutAnyOrigin, address_space=.GLOBAL
    ],
):
    """Load from a shared-memory TileTensor, store to global."""
    var val = src.load(coord[0, 0])
    dst.store(coord[0, 0], val)


def shared_store_kernel(
    dst: TileTensor[
        mut=True, .float32, _4x4, MutAnyOrigin, address_space=.SHARED
    ],
    src: TileTensor[.float32, _4x4, MutAnyOrigin, address_space=.GLOBAL],
):
    """Load from global, store to a shared-memory TileTensor."""
    var val = src.load(coord[0, 0])
    dst.store(coord[0, 0], val)


def invariant_load_kernel(
    src: TileTensor[.float32, _4x4, ImmutAnyOrigin, address_space=.GLOBAL],
    dst: TileTensor[
        mut=True, .float32, _4x4, MutAnyOrigin, address_space=.GLOBAL
    ],
):
    """Invariant load should produce a non-coherent (ld.global.nc) instruction.
    """
    var val = src.load[invariant=True](coord[0, 0])
    dst.store(coord[0, 0], val)


def vectorized_load_store_kernel(
    tensor: TileTensor[
        mut=True, .float32, _4x4, MutAnyOrigin, address_space=.GLOBAL
    ],
):
    """Width-4 load/store should produce vectorized (v4) instructions."""
    var val = tensor.load[width=4](coord[0, 0])
    tensor.store[width=4](coord[1, 0], val)


def atomic_add_kernel(
    tensor: TileTensor[
        mut=True,
        DType.int32,
        _4x4,
        MutAnyOrigin,
        address_space=AddressSpace.GLOBAL,
    ],
) raises:
    """Perform an atomic add through TileTensor to test lowering."""
    tensor.atomic_add(coord[0, 0], Scalar[DType.int32](1))


def atomic_max_kernel(
    tensor: TileTensor[
        mut=True,
        DType.int32,
        _4x4,
        MutAnyOrigin,
        address_space=AddressSpace.GLOBAL,
    ],
) raises:
    """Perform an atomic max through TileTensor to test lowering."""
    tensor.atomic_max(coord[0, 0], Scalar[DType.int32](0))


def atomic_compare_exchange_kernel(
    tensor: TileTensor[
        mut=True,
        DType.int32,
        _4x4,
        MutAnyOrigin,
        address_space=AddressSpace.GLOBAL,
    ],
) raises:
    """Perform an atomic compare-exchange through TileTensor to test lowering.
    """
    var expected = Scalar[DType.int32](0)
    _ = tensor.atomic_compare_exchange(
        coord[0, 0],
        expected,
        Scalar[DType.int32](1),
    )


# ===-----------------------------------------------------------------------===#
# Tests
# ===-----------------------------------------------------------------------===#


def test_tile_tensor_global_load_store() raises:
    """Global-memory TileTensor load/store emits ld.global and st.global."""
    var asm = _compile_code[global_load_store_kernel, target=_SM80]()
    assert_true(
        "ld.global" in asm,
        "expected ld.global for TileTensor global load",
    )
    assert_true(
        "st.global" in asm,
        "expected st.global for TileTensor global store",
    )


def test_tile_tensor_shared_load() raises:
    """Shared-memory TileTensor load emits ld.shared."""
    var asm = _compile_code[shared_load_kernel, target=_SM80]()
    assert_true(
        "ld.shared" in asm,
        "expected ld.shared for TileTensor shared-memory load",
    )


def test_tile_tensor_shared_store() raises:
    """Shared-memory TileTensor store emits st.shared."""
    var asm = _compile_code[shared_store_kernel, target=_SM80]()
    assert_true(
        "st.shared" in asm,
        "expected st.shared for TileTensor shared-memory store",
    )


def test_tile_tensor_invariant_load() raises:
    """Invariant TileTensor load emits ld.global.nc (non-coherent)."""
    var asm = _compile_code[invariant_load_kernel, target=_SM80]()
    assert_true(
        "ld.global.nc" in asm,
        "expected ld.global.nc for invariant TileTensor load",
    )


def test_tile_tensor_vectorized_load_store() raises:
    """Width-4 TileTensor load/store emits v4 vectorized instructions."""
    var asm = _compile_code[vectorized_load_store_kernel, target=_SM80]()
    assert_true(
        ".v4." in asm,
        "expected .v4. for vectorized TileTensor load/store",
    )


def test_tile_tensor_atomic_add() raises:
    """TileTensor atomic add emits an atomic add instruction."""
    var asm = _compile_code[atomic_add_kernel, target=_SM80]()
    assert_true(
        "atom.acquire.sys.global.add" in asm,
        "expected atom.acquire.sys.global.add for TileTensor atomic_add",
    )


def test_tile_tensor_atomic_max() raises:
    """TileTensor atomic max emits an atomic max instruction."""
    var asm = _compile_code[atomic_max_kernel, target=_SM80]()
    assert_true(
        "atom.acquire.sys.global.max" in asm,
        "expected atom.acquire.sys.global.max for TileTensor atomic_max",
    )


def test_tile_tensor_atomic_compare_exchange() raises:
    """TileTensor atomic compare_exchange emits a compare-and-swap instruction.
    """
    var asm = _compile_code[atomic_compare_exchange_kernel, target=_SM80]()
    assert_true(
        "atom.acquire.sys.global" in asm and ("cas" in asm or "cmpxchg" in asm),
        (
            "expected an atomic compare-exchange instruction for TileTensor"
            " atomic_compare_exchange"
        ),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
