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
"""Debug allocator poison pattern detection for uninitialized memory reads.

When compiled with `-D MOJO_STDLIB_SIMD_UNINIT_CHECK=true`, the helpers in
this module check loaded float values against the bit pattern written by
the debug allocator (`MODULAR_DEBUG_DEVICE_ALLOCATOR=uninitialized-poison`).
A match prints a diagnostic identifying the load site and the offending
lane, then triggers `abort()`. When disabled (the default), `@__parameter
if` / `comptime if` eliminates all checking code at compile time with zero
runtime overhead.

The poison pattern is the bit pattern of the largest finite value of the
float type (e.g. `FLT_MAX` for `float32`). This is intentionally non-NaN
and non-Inf so the poison fill does not conflict with the `nan-check`
debug pass when kernels leave allocation padding un-overwritten.
"""

from std.builtin.dtype import _unsigned_integral_type_of
from std._gpu import block_idx, thread_idx
from std.os import abort
from std.reflection import SourceLocation, call_location
from std.sys import is_apple_gpu, is_gpu
from std.sys.defines import get_defined_string
from std.utils.numerics import max_finite

comptime _UNINIT_CHECK_ENABLED = (
    get_defined_string["MOJO_STDLIB_SIMD_UNINIT_CHECK", "false"]() == "true"
)


# Float dtypes that participate in the poison check. The set must stay in
# sync with the dtypes that `Device::createDeviceMemory` poison-fills in
# `Driver.cpp`. Excluded:
# - `float4_e2m1fn`: sub-byte; the byte-granularity allocator fill can't
#   write a single 4-bit lane, so the C++ side skips the fill.
# - `float8_e3m4`, `float8_e8m0fnu`: `max_finite` is not defined for these
#   scale-only / no-mantissa formats, so the C++ side skips the fill too.
# - All other fp8 formats: the `max_finite` poison sentinel is the same
#   bit pattern produced by legitimate saturate-to-max in narrow fp8, so
#   the check yields pervasive false positives. The C++ side skips these
#   too.
@always_inline
def _is_poison_checked_dtype[dtype: DType]() -> Bool:
    return (
        dtype == .float16
        or dtype == .bfloat16
        or dtype == .float32
        or dtype == .float64
    )


@always_inline
def _poison_abort[
    uint_type: DType, //, dtype: DType
](
    poisoned_value: Scalar[uint_type],
    lane: Int,
    location: Optional[SourceLocation] = {},
) -> Never:
    """Reports a poison-pattern match and aborts.

    Unlike the standard `abort(msg)` path in `os.mojo`, this prints from
    *any* trapping lane rather than gating to thread (0,0,0) of block
    (0,0,0). When poison hits only a subset of lanes the (0,0,0) gate
    almost always silences the message entirely and the host sees only
    `CUDA_ERROR_LAUNCH_FAILED` with no actionable context. Printing
    unconditionally costs at most a handful of printf lines (one per
    trapping lane) and surfaces the kernel-source location, dtype, and
    SIMD lane that observed the poison.

    Parameters:
        uint_type: The unsigned-integral storage dtype of `dtype`; inferred
            from `poisoned_value`.
        dtype: The float dtype being loaded.

    Args:
        poisoned_value: The raw bit pattern observed on the failing lane.
        lane: Index within the SIMD vector that matched poison.
        location: Source location of the load that tripped the check; if
            unset, the caller's site (3 inline frames up: `_poison_abort`
            -> `_check_not_poison{,_masked}` -> the loading helper) is
            used.
    """
    # `call_location()` can't resolve in a parameter-context default expr,
    # so default to {} and resolve here in the function body.
    var loc = location.or_else(call_location[inline_count=3]())

    comptime if is_apple_gpu():
        # FIXME: Apple GPU printf path is broken (MOCO-3697); fall through
        # to a bare trap. The host still gets the trap, just no message.
        pass
    elif is_gpu():
        print(
            (
                t"UNINIT_READ at {loc}: dtype={dtype} lane={lane}"
                t" value={poisoned_value} block=[{block_idx.x},{block_idx.y},"
                t"{block_idx.z}] thread=[{thread_idx.x},{thread_idx.y},"
                t"{thread_idx.z}]: load matched debug allocator poison sentinel"
            ),
            flush=True,
        )
    else:
        print(
            (
                t"UNINIT_READ at {loc}: dtype={dtype} lane={lane}"
                t" value={poisoned_value}:"
                t" load matched debug allocator poison sentinel"
            ),
            flush=True,
        )

    abort()


@always_inline
def _check_not_poison[dtype: DType, width: Int](val: SIMD[dtype, width]):
    """Checks that a loaded SIMD value doesn't match debug allocator poison.

    Only active when compiled with `-D MOJO_STDLIB_SIMD_UNINIT_CHECK=true`. Zero cost otherwise.
    """

    comptime if not _UNINIT_CHECK_ENABLED:
        return

    comptime if _is_poison_checked_dtype[dtype]():
        comptime uint_type = _unsigned_integral_type_of[dtype]()
        comptime poison = max_finite[dtype]().to_bits[uint_type]()

        var bits = val.to_bits[uint_type]()
        for i in range(width):
            if bits[i] == poison:
                _poison_abort[dtype](poisoned_value=poison, lane=i)


@always_inline
def _check_not_poison_masked[
    dtype: DType, width: Int
](val: SIMD[dtype, width], mask: SIMD[.bool, width]):
    """Checks unmasked lanes of a SIMD value for debug allocator poison.

    Only active when compiled with `-D MOJO_STDLIB_SIMD_UNINIT_CHECK=true`. Zero cost otherwise.
    Masked-off lanes contain passthrough values and are not checked.
    """
    comptime if not _UNINIT_CHECK_ENABLED:
        return

    comptime if _is_poison_checked_dtype[dtype]():
        # Work at the integer-bit level to avoid ARM NaN canonicalization.
        # Float-level select can convert arbitrary NaN to canonical qNaN,
        # causing false positives in masked-off lanes.
        comptime uint_type = _unsigned_integral_type_of[dtype]()
        comptime poison = max_finite[dtype]().to_bits[uint_type]()

        var bits = val.to_bits[uint_type]()
        var safe_bits = mask.select(bits, SIMD[uint_type, width](0))
        for i in range(width):
            if safe_bits[i] == poison:
                _poison_abort[dtype](poisoned_value=poison, lane=i)
