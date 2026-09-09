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
"""Shared helpers for packing GPU kernel launch arguments.

These are implementation details of `DeviceFunction`'s launch paths (see
`device_context.mojo` and `_device_context_metal.mojo`); they are not part of
the public API.
"""

from std.collections.optional import Optional


@always_inline
def _compact_zero_sized_capture_slots(
    dense_args_addrs: Pointer[OpaquePointer[MutAnyOrigin], MutUntrackedOrigin],
    capture_sizes: Pointer[UInt64, ImmUntrackedOrigin],
    num_leading_args: Int,
    num_captures: Int,
    dense_args_sizes: Optional[Pointer[UInt64, MutUntrackedOrigin]] = None,
) -> Int:
    """Compacts the capture slots of a packed launch-argument array in place,
    dropping zero-sized captures.

    NOTE(MOCO-4233): This is a hack resulting from legacy closures. A
    fully-static-layout capture (e.g. a `TileTensor`'s layout struct) is
    emitted as a zero-sized cross-device capture, but the device kernel
    elides it from its declared parameter list. The driver reads the packed
    argument array positionally against that parameter list, so a zero-sized
    slot preceding a real argument shifts every following argument by one:
    the kernel dereferences the wrong value (a hard illegal-address fault on
    CUDA), while Metal validates argument sizes and rejects any `== 0`.
    Dropping the zero-sized slots keeps the packed array aligned with the
    device kernel's parameter order.

    The capture slots start at `dense_args_addrs[num_leading_args]` (as
    written by the compiler-generated `populate` function) and are compacted
    in place, preserving relative order. Slots at and beyond the returned
    count are left stale and must not be read.

    Args:
        dense_args_addrs: The packed per-argument value pointers:
            `num_leading_args` argument slots followed by `num_captures`
            capture slots.
        capture_sizes: The store size in bytes of each capture, as recorded
            by the compiler.
        num_leading_args: The number of non-capture argument slots preceding
            the captures.
        num_captures: The number of capture slots.
        dense_args_sizes: Optional parallel per-argument sizes array; when
            provided, the surviving captures' sizes are written to it in
            lockstep with the compacted address slots.

    Returns:
        The effective argument count — `num_leading_args` plus the number of
        non-zero-sized captures. This is the slot count the device kernel
        actually reads, and must be the count passed to the launch call.
    """
    var effective_argc = num_leading_args
    for i in range(num_captures):
        if capture_sizes[unsafe_offset=i] != 0:
            dense_args_addrs[unsafe_offset=effective_argc] = dense_args_addrs[
                unsafe_offset=num_leading_args + i
            ]
            if dense_args_sizes:
                dense_args_sizes.value()[
                    unsafe_offset=effective_argc
                ] = capture_sizes[unsafe_offset=i]
            effective_argc += 1
    return effective_argc
