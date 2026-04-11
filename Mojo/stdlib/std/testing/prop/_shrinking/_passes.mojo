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

from std.sys.info import simd_width_of

from ._shrinker import ShrinkOracle, Stream, StreamChunk


# TODO: add larger block sizes and handle mulitples of simd width
def block_sizes() -> List[Int]:
    var l = List[Int]()
    var start = simd_width_of[DType.uint64]()

    while start > 0:
        l.append(start)
        start >>= 1

    return l^


def delete_chunk(stream: Stream, start: Int, chunk_size: Int) -> Stream:
    var cpy = Stream()
    for j in range(start):
        cpy.append(stream[j])
    for j in range(start + chunk_size, len(stream)):
        cpy.append(stream[j])
    return cpy^


def binary_search_value[
    O: ShrinkOracle, //
](mut oracle: O, i: Int, var hi: UInt64, mut improved: Bool):
    """Binary search the smallest value at index `i` that is still interesting.
    """
    var lo = UInt64(0)
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        var cpy = oracle.best_stream().copy()
        cpy[i] = mid

        if oracle.try_stream(cpy):
            improved = True
            # best was updated by try_stream, use its value
            hi = oracle.best_stream()[i]
        else:
            if mid == lo:
                break
            lo = mid


def multi_element_pass[
    O: ShrinkOracle,
    //,
    op: def[width: Int, //](StreamChunk[width]) thin -> StreamChunk[width],
](mut oracle: O) -> Bool:
    comptime for size in block_sizes():
        if size <= len(oracle.best_stream()):
            for i in range(0, len(oracle.best_stream()) - size + 1, size):
                var cpy = oracle.best_stream().copy()
                var ptr = cpy.unsafe_ptr().unsafe_offset(i)
                ptr.unsafe_store[width=size](
                    0, op(ptr.unsafe_load[width=size]())
                )

                if oracle.try_stream(cpy):
                    return True

    return False


trait ShrinkOp(Movable):
    def run[O: ShrinkOracle, //](mut self, mut oracle: O) -> Bool:
        ...


@fieldwise_init
struct DeleteChunkPass(Movable, ShrinkOp):
    """Try to delete contiguous chunks from the stream, largest first."""

    def run[O: ShrinkOracle, //](mut self, mut oracle: O) -> Bool:
        var n = len(oracle.best_stream())
        var chunk_size = n - 1

        while chunk_size >= 1:
            for i in range(n - chunk_size + 1):
                var candidate = delete_chunk(
                    oracle.best_stream(), i, chunk_size
                )
                if oracle.try_stream(candidate):
                    return True

            chunk_size -= 1

        return False


@fieldwise_init
struct ZeroChunkPass(Movable, ShrinkOp):
    """Try to zero out contiguous chunks of the stream."""

    def run[O: ShrinkOracle, //](mut self, mut oracle: O) -> Bool:
        @always_inline
        def zero[width: Int, //](s: StreamChunk[width]) -> StreamChunk[width]:
            return 0

        return multi_element_pass[zero](oracle)


@fieldwise_init
struct ReduceChunkPass(Movable, ShrinkOp):
    """Binary-search each individual element down to its smallest interesting
    value."""

    def run[O: ShrinkOracle, //](mut self, mut oracle: O) -> Bool:
        var improved = False

        # TODO: Use SIMD/chunked for this as well
        var n = len(oracle.best_stream())
        for i in range(n):
            ref current = oracle.best_stream()
            var hi = current[i]
            if hi == 0:
                continue

            binary_search_value(oracle, i, hi, improved)

            if improved:
                break

        return improved


@fieldwise_init
struct SortChunkPass(Movable, ShrinkOp):
    """TODO: Try and find a smaller ordering of the existing data."""

    def run[O: ShrinkOracle, //](mut self, mut oracle: O) -> Bool:
        return False


@fieldwise_init
struct RedistributePass(Movable, ShrinkOp):
    """TODO: Adjust adjacent values while keeping their sum constant."""

    def run[O: ShrinkOracle, //](mut self, mut oracle: O) -> Bool:
        return False


@fieldwise_init
struct SwapChunkPass(Movable, ShrinkOp):
    """TODO: Swap adjacent chunks."""

    def run[O: ShrinkOracle, //](mut self, mut oracle: O) -> Bool:
        return False
