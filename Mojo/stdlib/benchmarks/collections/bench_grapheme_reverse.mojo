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
"""Reverse grapheme iteration benchmarks.

Measures the impact of caching the reverse-iteration safe-start across
calls. Forward iteration is included as a reference baseline.
"""

from std.os import abort
from std.pathlib import _dir_of_current_file
from std.sys import stderr

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep


def make_string[
    length: Int = 0
](filename: String = "UN_charter_EN.txt") -> String:
    try:
        var directory = _dir_of_current_file() / "data"
        var f = open(directory / filename, "r")

        comptime if length == 0:
            return String(unsafe_from_utf8=f.read_bytes())

        var full = f.read_bytes()
        var items = List[Byte](capacity=length + len(full))
        while len(items) < length:
            items.extend(full.copy())
        var cut = length
        while cut < len(items) and (items[cut] & 0xC0) == 0x80:
            cut -= 1
        while len(items) > cut:
            _ = items.pop()
        return String(unsafe_from_utf8=items)
    except e:
        print(e, file=stderr)
    abort(String())


def bench_grapheme_iter_forward[
    length: Int, filename: StaticString
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    def call_fn() {imm}:
        var count = 0
        for _ in black_box(items).graphemes():
            count += 1
        keep(count)

    b.iter(call_fn)


def bench_grapheme_iter_reversed[
    length: Int, filename: StaticString
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    def call_fn() {imm}:
        var count = 0
        for _ in black_box(items).graphemes_reversed():
            count += 1
        keep(count)

    b.iter(call_fn)


def bench_grapheme_iter_alternating[
    length: Int, filename: StaticString
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    def call_fn() {imm}:
        var count = 0
        var iter = black_box(items).graphemes()
        while True:
            var f = iter.next()
            if not f:
                break
            count += 1
            var bk = iter.next_back()
            if not bk:
                break
            count += 1
        keep(count)

    b.iter(call_fn)


def main() raises:
    var m = Bench(BenchConfig(num_repetitions=1))
    comptime filenames = (
        StaticString("UN_charter_EN"),
        StaticString("UN_charter_AR"),
        StaticString("UN_charter_zh-CN"),
    )

    comptime lengths = (1_000, 10_000, 100_000)

    comptime for i in range(len(lengths)):
        comptime length = rebind[Int](lengths[i])

        comptime for j in range(len(filenames)):
            comptime fname = rebind[StaticString](filenames[j])
            comptime suffix = String("[", length, ",", fname, "]")
            m.bench_function(
                bench_grapheme_iter_forward[length, fname],
                BenchId(String("forward", suffix)),
            )
            m.bench_function(
                bench_grapheme_iter_reversed[length, fname],
                BenchId(String("reversed", suffix)),
            )
            m.bench_function(
                bench_grapheme_iter_alternating[length, fname],
                BenchId(String("alternating", suffix)),
            )

    print("")
    for info in m.info_vec:
        print(info.name, info.result.mean("ms"), sep=", ")
