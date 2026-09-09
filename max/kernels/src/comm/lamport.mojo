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
"""Shared primitives for the barrier-free Lamport communication protocol.

A Lamport-style exchange replaces an explicit cross-GPU readiness flag with a
reserved value embedded *inside* the data pack: negative zero. Because IEEE-754
defines `-0.0 == +0.0` and `x + (-0.0) == x`, a producer can sanitize its real
data so that no lane ever holds the `-0.0` bit pattern, and a consumer can then
treat "this pack still contains a `-0.0` lane" as "the producer has not written
here yet." The 16-byte single-copy-atomic store couples "data present" with
"sentinel gone", so no separate flag word and no hot-path atomic are required.

The three functions here are the dtype- and width-parameterized building blocks
of that protocol, with no allreduce-specific logic so broadcast and other comm
kernels can reuse them:

- `remove_neg_zero`: producer-side sanitize (map `-0.0` lanes to `+0.0`).
- `has_neg_zero`: consumer-side readiness poll (any lane still the sentinel).
- `set_neg_zero`: produce the all-sentinel "not ready" pack.

`LamportGeneration` owns the three-buffer generation rotation (`flag % 3`) that
defends against the classic Lamport write-after-read hazard, so the kernel body
and any future consumer share identical generation semantics.

See the internal Lamport allreduce design doc, section
"Design > 1. Shared sentinel primitive", for the protocol context.
"""

from std.memory import bitcast
from std.sys import bit_width_of, size_of
from std.builtin.dtype import _unsigned_integral_type_of


comptime LAMPORT_SENTINEL_U32 = UInt32(0x80000000)
"""The universal Lamport "not ready" sentinel, as a `uint32` (fp32 `-0.0`)."""


struct Lamport:
    """Sizing and state-block layout for the barrier-free Lamport comm region.

    The single source of truth these values derive from: `Signal` (sync.mojo)
    sizes its embedded region from here, and the kernel reads the pack/state
    layout from here, instead of restating the arithmetic at each site. Holds
    only what is independent of `Signal` and `MAX_GPUS` (so `lamport.mojo` keeps
    no dependency on `sync.mojo`); `Signal` composes the full region size from
    `MAX_SMALL_MESSAGE_BYTES` x `MAX_GPUS` x `LamportGeneration.NUM_GENERATIONS`.
    """

    comptime MAX_SMALL_MESSAGE_BYTES = 1024 * 1024
    """Largest per-rank message a Lamport generation slot reserves for (bytes) --
    the small/large dispatch ceiling. 1 MiB is the measured B200 crossover."""

    comptime ATOMIC_BYTES = 16
    """The 128-bit single-copy-atomic pack width the sentinel protocol needs."""

    comptime MAX_PACKS = Self.MAX_SMALL_MESSAGE_BYTES // Self.ATOMIC_BYTES
    """`MAX_SMALL_MESSAGE_BYTES` in 128-bit packs; the per-slot generation stride.
    """

    # Layout of the `Signal.lamport_state` block:
    # [flag, prev_num_elements, arrival, reserved].
    comptime STATE_LEN = 4
    comptime STATE_FLAG = 0
    comptime STATE_PREV_ELEMS = 1
    comptime STATE_ARRIVAL = 2


@always_inline
def _sentinel_bits[
    dtype: DType
]() -> Scalar[_unsigned_integral_type_of[dtype]()]:
    """Returns the per-lane raw-bit negative-zero sentinel for `dtype`.

    Negative zero sets only the sign bit, so the sentinel is `1 << (width - 1)`:
    `0x8000` for 16-bit (bf16/fp16) and `0x80000000` for fp32.

    Parameters:
        dtype: The transport float type. Must be bfloat16, float16, or float32.

    Returns:
        The negative-zero bit pattern as the matching unsigned-integer scalar.
    """
    comptime assert (
        dtype == .bfloat16 or dtype == .float16 or dtype == .float32
    ), (
        "Lamport sentinel supports bfloat16, float16, and float32 only; fp8"
        " transport is out of scope (>8 lanes/pack and the coarse sentinel"
        " check make it unsafe)"
    )
    comptime uint = _unsigned_integral_type_of[dtype]()
    # 1 << (bitwidth - 1): only the sign bit set == negative zero.
    return Scalar[uint](1) << Scalar[uint](bit_width_of[uint]() - 1)


@always_inline
def remove_neg_zero[
    dtype: DType, width: Int
](v: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Maps every `-0.0` lane to `+0.0`, leaving all other lanes bit-identical.

    This is the producer-sanitize step: after it, real data can never be
    mistaken for the readiness sentinel. Lanes holding `+0.0`, NaN, Inf, or any
    normal value are returned bit-for-bit unchanged. The implementation is
    branchless: bitcast to the unsigned lane type, `select` the sentinel lanes
    to `+0.0` bits (all zero), and bitcast back.

    Parameters:
        dtype: The transport float type. Must be bfloat16, float16, or float32.
        width: The SIMD pack width.

    Args:
        v: The pack to sanitize.

    Returns:
        `v` with any `-0.0` lane replaced by `+0.0`.
    """
    comptime uint = _unsigned_integral_type_of[dtype]()
    var bits = bitcast[uint, width](v)
    var is_neg_zero = bits.eq(SIMD[uint, width](_sentinel_bits[dtype]()))
    # +0.0 is all-zero bits; map sentinel lanes there, keep the rest.
    var sanitized = is_neg_zero.select(SIMD[uint, width](0), bits)
    return bitcast[dtype, width](sanitized)


@always_inline
def has_neg_zero[dtype: DType, width: Int](v: SIMD[dtype, width]) -> Bool:
    """Returns True if any lane of `v` holds the `-0.0` sentinel bit pattern.

    This is the consumer-side readiness poll: a pack that still contains a
    sentinel lane has not been fully written by the producer. The any-lane
    reduction also defends against a torn read that leaves a single sentinel
    lane behind a partially-visible store.

    Parameters:
        dtype: The transport float type. Must be bfloat16, float16, or float32.
        width: The SIMD pack width.

    Args:
        v: The pack to test.

    Returns:
        True if at least one lane equals the sentinel, False otherwise.
    """
    comptime uint = _unsigned_integral_type_of[dtype]()
    var bits = bitcast[uint, width](v)
    return Bool(bits.eq(SIMD[uint, width](_sentinel_bits[dtype]())).reduce_or())


@always_inline
def set_neg_zero[dtype: DType, width: Int]() -> SIMD[dtype, width]:
    """Returns the universal "not ready" sentinel pack.

    The pack is the fp32 `-0.0` bit pattern (`0x80000000`) tiled over every 4
    bytes. This one pattern is DTYPE-INDEPENDENT, yet `has_neg_zero[dtype]`
    detects it as the sentinel for every supported transport dtype, because:

    - the readiness poll is ANY-lane (it spins while *any* lane is `-0.0`), and
    - IEEE floats put the sign bit at the element MSB, so `0x80` every 4 bytes
      lands on a sign bit for fp32 (all lanes) and for the odd lanes of 2-byte
      dtypes (bf16/fp16) -- at least one `-0.0` lane under each interpretation.

    Parameters:
        dtype: The transport float type (bfloat16, float16, or float32).
        width: The SIMD pack width. `width * sizeof(dtype)` must be a multiple
            of 4 bytes (true for every 128-bit pack used by the protocol).

    Returns:
        The sentinel pack reinterpreted as `SIMD[dtype, width]`.
    """
    comptime pack_bytes = width * size_of[dtype]()
    comptime assert (
        pack_bytes % 4 == 0
    ), "Lamport sentinel pack must be a multiple of 4 bytes"
    comptime n_u32 = pack_bytes // 4
    return bitcast[dtype, width](SIMD[.uint32, n_u32](LAMPORT_SENTINEL_U32))


struct LamportGeneration:
    """Owns the three-buffer generation rotation for a Lamport exchange.

    The classic Lamport hazard is a fast rank racing into call N+1 and
    overwriting a slot a slow rank has not yet read for call N. Three rotating
    buffer generations defend against it: each call writes-and-reads one
    generation while clearing the generation that will be reused two calls
    later, leaving exactly one generation of skew slack.

    Given a monotonically-increasing call counter `flag`:

    - `data_index` = `flag % 3` is the generation written and read this call;
    - `clear_index` = `(flag + 2) % 3` is the generation cleared back to the
      sentinel this call (it will next be written two calls from now).

    `data_index != clear_index` always holds, which is the one-generation-of-
    slack invariant. This mirrors `circular_add`'s pure-integer style in
    `sync.mojo` so the kernel body and any future broadcast consumer share
    identical semantics.
    """

    comptime NUM_GENERATIONS = 3
    """Number of rotating buffer generations: read-this-call +
    write-next-call + clear-for-the-call-after."""

    @staticmethod
    @always_inline
    def data_index(flag: Int) -> Int:
        """Returns the generation index written and read on this call.

        Args:
            flag: The monotonically-increasing call counter.

        Returns:
            `flag % NUM_GENERATIONS`.
        """
        return flag % Self.NUM_GENERATIONS

    @staticmethod
    @always_inline
    def clear_index(flag: Int) -> Int:
        """Returns the generation index cleared to the sentinel on this call.

        This is the generation that will next be written two calls from now;
        clearing it now overlaps with the current exchange.

        Args:
            flag: The monotonically-increasing call counter.

        Returns:
            `(flag + NUM_GENERATIONS - 1) % NUM_GENERATIONS`, i.e. `(flag + 2)
            % 3` for the default of three generations.
        """
        return (flag + Self.NUM_GENERATIONS - 1) % Self.NUM_GENERATIONS
