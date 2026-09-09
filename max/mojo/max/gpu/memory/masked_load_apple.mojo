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
"""Apple M5 (AGX3) hardware edge-masked vector loads.

`build_edge_mask` computes a per-lane in-bounds mask; the `*edge_masked_load`
family does a predicated vector load that zeroes masked-off lanes, letting a
kernel vectorize a boundary tile without a scalar remainder loop or an
out-of-bounds read. The three entry points differ only in address-space
resolution:

- `edge_masked_load` — the pointer's own space (must be `GLOBAL` or `SHARED`).
- `gmem_edge_masked_load` — reinterprets any pointer as device (global).
- `smem_edge_masked_load` — reinterprets any pointer as threadgroup (shared).

Apple M5 only: the AGX3 intrinsics don't exist elsewhere, so callers dispatch
on `compute_capability() == 5`.
"""

from std.collections.string import StaticString
from std.ffi import external_call
from std.memory import bitcast
from std.sys import size_of


@always_inline
def build_edge_mask(
    index: Int32, lower_bound: Int32, upper_bound: Int32
) -> Int16:
    """Computes the per-lane edge mask for a vector access (Apple M5 only).

    Bit `i` of the result is set when element `index + i` lies in
    `[lower_bound, upper_bound)`. Pass it as an `*edge_masked_load` `mask`.
    Emits the AGX3 `edgecheck` instruction.

    Args:
        index: Linear element index of lane 0.
        lower_bound: Inclusive lower bound of the valid range.
        upper_bound: Exclusive upper bound of the valid range.

    Returns:
        A 16-bit per-lane mask (up to 4 lanes).
    """
    return external_call["llvm.agx3.edgecheck", Int16](
        index, lower_bound, upper_bound
    )


@always_inline
def _emask_load[
    dtype: DType,
    src_space: AddressSpace,
    //,
    width: Int,
    name_space: StaticString,
    target_space: AddressSpace,
](
    ptr: Pointer[Scalar[dtype], _, address_space=src_space],
    mask: Int16,
) -> SIMD[dtype, width]:
    """Shared predicated-load implementation for the `*edge_masked_load` family.

    Reinterprets `ptr` into `target_space` and emits the
    `llvm.agx3.load.with.emask.<name_space>.vNiM` intrinsic.
    """
    comptime assert (
        width == 1 or width == 2 or width == 4
    ), "edge-masked load width must be 1, 2, or 4"
    comptime bytes = size_of[dtype]()
    comptime assert (
        bytes == 1 or bytes == 2 or bytes == 4 or bytes == 8
    ), "edge-masked load requires an 8-, 16-, 32-, or 64-bit dtype"

    comptime bits = bytes * 8
    comptime vec = StaticString("v1") if width == 1 else (
        StaticString("v2") if width == 2 else StaticString("v4")
    )
    comptime elem = StaticString("i8") if bits == 8 else (
        StaticString("i16") if bits
        == 16 else (StaticString("i32") if bits == 32 else StaticString("i64"))
    )
    comptime name = (
        StaticString("llvm.agx3.load.with.emask.")
        + name_space
        + "."
        + vec
        + elem
    )

    # Intrinsic takes a byte pointer and returns raw integer lanes.
    comptime lane_dtype = DType.uint8 if bits == 8 else (
        DType.uint16 if bits
        == 16 else (DType.uint32 if bits == 32 else DType.uint64)
    )
    comptime full_mask = Int16((1 << width) - 1)
    comptime elt_size = Int16(size_of[dtype]())

    var byte_ptr = ptr.unsafe_bitcast[UInt8]().unsafe_address_space_cast[
        target_space
    ]()
    var raw = external_call[name, SIMD[lane_dtype, width]](
        byte_ptr, mask, full_mask, elt_size
    )
    return bitcast[dtype, width](raw)


@always_inline
def edge_masked_load[
    dtype: DType,
    src_space: AddressSpace,
    //,
    width: Int,
](
    ptr: Pointer[Scalar[dtype], _, address_space=src_space],
    mask: Int16,
) -> SIMD[dtype, width]:
    """Loads a predicated (edge-masked) vector in the pointer's own address
    space (Apple M5 only).

    Loads `width` elements from `ptr`, zeroing any lane whose `mask` bit is
    clear; masked-off lanes are not read, so the vector may safely straddle the
    end of a buffer. For a generic-typed pointer use `gmem_edge_masked_load` /
    `smem_edge_masked_load`. Emits the AGX3 `load.with.emask` instruction; build
    `mask` with `build_edge_mask`.

    Parameters:
        dtype: Element type of the pointer (inferred).
        src_space: Pointer address space (inferred).
        width: Number of elements to load.

    Args:
        ptr: Base pointer of the load.
        mask: Per-lane active mask, e.g. from `build_edge_mask`.

    Returns:
        A `SIMD[dtype, width]`; in-range lanes loaded, masked-off lanes zero.

    Constraints:
        `width` in {1, 2, 4}, `dtype` 8/16/32/64-bit, `src_space` `GLOBAL` or
        `SHARED`. Apple M5 only.
    """
    comptime assert (
        src_space == AddressSpace.GLOBAL or src_space == AddressSpace.SHARED
    ), (
        "edge_masked_load requires a GLOBAL or SHARED pointer; use"
        " gmem_edge_masked_load / smem_edge_masked_load to reinterpret a"
        " generic pointer"
    )
    comptime name_space = StaticString(
        "global"
    ) if src_space == AddressSpace.GLOBAL else StaticString("local")
    return _emask_load[width, name_space, src_space](ptr, mask)


@always_inline
def gmem_edge_masked_load[
    dtype: DType,
    src_space: AddressSpace,
    //,
    width: Int,
](
    ptr: Pointer[Scalar[dtype], _, address_space=src_space],
    mask: Int16,
) -> SIMD[dtype, width]:
    """Loads an edge-masked vector, treating `ptr` as device (global) memory
    (Apple M5 only).

    Like `edge_masked_load` but reinterprets any pointer as global — for a
    generic-typed pointer to device memory. Emits AGX3
    `load.with.emask.global`.

    Parameters:
        dtype: Element type of the pointer (inferred).
        src_space: Pointer source address space (inferred).
        width: Number of elements to load.

    Args:
        ptr: Base pointer of the load, pointing to global memory.
        mask: Per-lane active mask, e.g. from `build_edge_mask`.

    Returns:
        A `SIMD[dtype, width]`; in-range lanes loaded, masked-off lanes zero.

    Constraints:
        `width` in {1, 2, 4}, `dtype` 8/16/32/64-bit. Apple M5 only.
    """
    return _emask_load[width, "global", AddressSpace.GLOBAL](ptr, mask)


@always_inline
def smem_edge_masked_load[
    dtype: DType,
    src_space: AddressSpace,
    //,
    width: Int,
](
    ptr: Pointer[Scalar[dtype], _, address_space=src_space],
    mask: Int16,
) -> SIMD[dtype, width]:
    """Loads an edge-masked vector, treating `ptr` as threadgroup (shared)
    memory (Apple M5 only).

    Like `edge_masked_load` but reinterprets any pointer as shared — for a
    generic-typed pointer to threadgroup memory. Emits AGX3
    `load.with.emask.local`.

    Parameters:
        dtype: Element type of the pointer (inferred).
        src_space: Pointer source address space (inferred).
        width: Number of elements to load.

    Args:
        ptr: Base pointer of the load, pointing to threadgroup memory.
        mask: Per-lane active mask, e.g. from `build_edge_mask`.

    Returns:
        A `SIMD[dtype, width]`; in-range lanes loaded, masked-off lanes zero.

    Constraints:
        `width` in {1, 2, 4}, `dtype` 8/16/32/64-bit. Apple M5 only.
    """
    return _emask_load[width, "local", AddressSpace.SHARED](ptr, mask)
