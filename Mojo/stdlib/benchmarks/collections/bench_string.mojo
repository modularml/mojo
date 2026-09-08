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

from std.collections import Optional
from std.collections.string._utf8 import _is_valid_utf8
from std.collections.string.string_span import _split
from std.os import abort
from std.pathlib import _dir_of_current_file
from std.random import seed
from std.sys import stderr

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
def make_string[
    length: Int = 0
](filename: String = "UN_charter_EN.txt") -> String:
    """Make a `String` made of items in the `./data` directory.

    Parameters:
        length: The length in bytes of the resulting `String`. If == 0 -> the
            whole file content.

    Args:
        filename: The name of the file inside the `./data` directory.
    """

    try:
        var directory = _dir_of_current_file() / "data"
        var f = open(directory / filename, "r")

        comptime if length > 0:
            var items = f.read_bytes(length)
            var i = 0
            while length > len(items):
                items.append(items[i])
                i = i + 1 if i < len(items) - 1 else 0
            # `length` can truncate mid-codepoint; drop trailing bytes until
            # the buffer ends on a valid UTF-8 boundary again.
            while len(items) > 0 and not _is_valid_utf8(Span(items)):
                _ = items.pop()
            return String(unsafe_from_utf8=items)
        else:
            return String(unsafe_from_utf8=f.read_bytes())
    except e:
        print(e, file=stderr)
    abort(String())


# ===-----------------------------------------------------------------------===#
# Benchmark string init
# ===-----------------------------------------------------------------------===#
def bench_string_init(mut b: Bencher) raises:
    @always_inline
    def call_fn():
        for _ in range(1000):
            var string = String()
            keep(string)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string count
# ===-----------------------------------------------------------------------===#
def bench_string_count[
    length: Int = 0,
    filename: StaticString = "UN_charter_EN",
    sequence: StaticString = "a",
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    def call_fn() {imm}:
        var amnt = black_box(items).count(black_box(sequence))
        keep(amnt)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string split
# ===-----------------------------------------------------------------------===#
def bench_string_split[
    length: Int = 0,
    filename: StaticString = "UN_charter_EN",
    sequence: Optional[StaticString] = None,
](mut b: Bencher) raises:
    var items = StringSlice(make_string[length](filename + ".txt")).as_imm()

    @always_inline
    def call_fn() {imm}:
        var res: List[type_of(items)]

        comptime if sequence:
            res = _split[has_maxsplit=False](
                black_box(items), black_box(sequence.value()), black_box(-1)
            )
        else:
            res = _split[has_maxsplit=False](
                black_box(items), None, black_box(-1)
            )
        keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string split dense
# ===-----------------------------------------------------------------------===#
def bench_string_split_dense[count: Int](mut b: Bencher) raises:
    """Benchmark split with dense separators producing many items.

    This measures the allocation strategy when split produces many items
    (e.g. CSV-like data with dense separators).
    """
    # Build a string "item,item,item,..." with `count` items.
    var buf = String(capacity_bytes=count * 5)
    for i in range(count):
        if i > 0:
            buf += ","
        buf += "item"

    @always_inline
    def call_fn() {imm}:
        var s = StringSlice(buf).as_imm()
        var res = _split[has_maxsplit=False](
            black_box(s), black_box(StaticString(",")), black_box(-1)
        )
        keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string join
# ===-----------------------------------------------------------------------===#
def bench_string_join[short: Bool](mut b: Bencher) raises:
    var count: Int
    comptime if short:
        count = 100
    else:
        count = 1000

    var word_list = List[String](capacity=count)
    for i in range(count):
        word_list.append(String(i))

    var separator = String(",")

    @always_inline
    def call_fn() {imm}:
        for _ in range(1_000):
            var res = black_box(separator).join(black_box(word_list))
            keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string splitlines
# ===-----------------------------------------------------------------------===#
def bench_string_splitlines[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = StringSlice(make_string[length](filename + ".txt"))

    @always_inline
    def call_fn() {imm}:
        for _ in range(1_000_000 // length):
            var res = black_box(items).splitlines()
            keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string lower
# ===-----------------------------------------------------------------------===#
def bench_string_lower[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    def call_fn() {imm}:
        var res = black_box(items).lower()
        keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string upper
# ===-----------------------------------------------------------------------===#
def bench_string_upper[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    def call_fn() {imm}:
        var res = black_box(items).upper()
        keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string replace
# ===-----------------------------------------------------------------------===#
def bench_string_replace[
    length: Int = 0,
    filename: StaticString = "UN_charter_EN",
    old: StaticString = "a",
    new: StaticString = "A",
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    def call_fn() {imm}:
        var res = black_box(items).replace(black_box(old), black_box(new))
        keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string count_codepoints
# ===-----------------------------------------------------------------------===#
def bench_string_count_codepoints[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    def call_fn() {imm}:
        var res = black_box(items).count_codepoints()
        keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string find single
# ===-----------------------------------------------------------------------===#
def bench_string_find_single[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    def call_fn() {imm}:
        # this is to help with instability when measuring small strings
        for _ in range(10**6 // length):
            var res = black_box(items).find(
                black_box("Z")
            )  # something that probably won't be there
            keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string find multiple
# ===-----------------------------------------------------------------------===#
def bench_string_find_multiple[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")
    var sequence = "ZZZZ"  # something that probably won't be there

    @always_inline
    def call_fn() {imm}:
        # this is to help with instability when measuring small strings
        for _ in range(10**6 // length):
            var res = black_box(items).find(black_box(sequence))
            keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string _is_valid_utf8
# ===-----------------------------------------------------------------------===#
def bench_string_is_valid_utf8[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".html")

    @always_inline
    def call_fn() {imm}:
        var res = _is_valid_utf8(black_box(items).as_bytes())
        keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark write_utf8
# ===-----------------------------------------------------------------------===#
def bench_write_utf8[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")
    var codepoints_iter = items.codepoints()
    # appending to a list to avoid paying the overhead of codepoint parsing
    var codepoints = List[Codepoint](capacity=len(codepoints_iter))
    for c in codepoints_iter:
        codepoints.append(c)

    @always_inline
    def call_fn() {imm}:
        var data = Array[Byte, 4](uninitialized=True)
        # this is to help with instability when measuring small strings
        for _ in range(10**6 // length):
            for i in range(len(codepoints)):
                var res = black_box(codepoints.unsafe_get(i)).unsafe_write_utf8(
                    black_box(data).unsafe_ptr()
                )
                keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string write
# ===-----------------------------------------------------------------------===#
def bench_string_write[short: Bool](mut b: Bencher) raises:
    var items = make_string[1000]("UN_charter_EN.txt")
    # workaround for "allows writing to mem location ..."
    # even though I tried using an immutable StringSlice
    var items_2 = items.copy()
    var items_3 = items.copy()
    var items_4 = items.copy()
    var items_5 = items.copy()

    @always_inline
    def call_fn() {imm}:
        for _ in range(1_000_000):
            var res: String

            comptime if short:  # less than 24 bytes
                res = String(
                    black_box(0),
                    black_box(" is "),
                    black_box("a"),
                    black_box(String(" number")),
                )
            else:  # 5001 bytes long
                res = String(
                    black_box(0),
                    black_box(items),
                    black_box(items_2),
                    black_box(items_3),
                    black_box(items_4),
                    black_box(items_5),
                )
            keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string repr
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct NullWriter(ImplicitlyCopyable, Writer):
    def write_string(mut self, string: StringSlice):
        keep(string)


def bench_string_repr[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher):
    var items = make_string[length](filename + ".txt")

    @always_inline
    def call_fn() {imm}:
        # this is to help with instability when measuring small strings
        for _ in range(10**6 // length):
            var writer = NullWriter()
            black_box(items).write_repr_to(writer)
            keep(writer)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main() raises:
    seed()
    var m = Bench(BenchConfig(num_repetitions=1))
    comptime filenames = (
        StaticString("UN_charter_EN"),
        StaticString("UN_charter_ES"),
        StaticString("UN_charter_AR"),
        StaticString("UN_charter_RU"),
        StaticString("UN_charter_zh-CN"),
    )
    comptime old_chars = (
        StaticString("a"),
        StaticString("ó"),
        StaticString("ل"),
        StaticString("и"),
        StaticString("一"),
    )
    comptime new_chars = (
        StaticString("A"),
        StaticString("Ó"),
        StaticString("ل"),
        StaticString("И"),
        StaticString("一"),
    )

    comptime lengths = (10, 30, 50, 100, 1000, 10_000, 100_000, 1_000_000)
    """At an average 5 letters per word and 300 words per page
    (in the English language):

    - 10: 2 words
    - 30: 6 words
    - 50: 10 words
    - 100: 20 words
    - 1000: ~ 1/2 page (200 words)
    - 10_000: ~ 7 pages (2k words)
    - 100_000: ~ 67 pages (20k words)
    - 1_000_000: ~ 667 pages (200k words)
    """

    m.bench_function(bench_string_init, BenchId("bench_string_init"))
    m.bench_function(
        bench_string_write[True], BenchId(String("bench_string_write_short"))
    )
    m.bench_function(
        bench_string_write[False], BenchId(String("bench_string_write_long"))
    )

    comptime for i in range(len(lengths)):
        comptime length = rebind[Int](lengths[i])

        comptime for j in range(len(filenames)):
            comptime fname = rebind[StaticString](filenames[j])
            comptime old = rebind[StaticString](old_chars[j])
            comptime new = rebind[StaticString](new_chars[j])
            comptime suffix = String("[", length, "]")  # "(" + fname + ")"
            m.bench_function(
                bench_string_count[length, fname, old],
                BenchId(String("bench_string_count", suffix)),
            )
            m.bench_function(
                bench_string_split[length, fname, old],
                BenchId(String("bench_string_split", suffix)),
            )
            m.bench_function(
                bench_string_split[length, fname],
                BenchId(String("bench_string_split_none", suffix)),
            )
            m.bench_function(
                bench_string_splitlines[length, fname],
                BenchId(String("bench_string_splitlines", suffix)),
            )
            m.bench_function(
                bench_string_lower[length, fname],
                BenchId(String("bench_string_lower", suffix)),
            )
            m.bench_function(
                bench_string_upper[length, fname],
                BenchId(String("bench_string_upper", suffix)),
            )
            m.bench_function(
                bench_string_replace[length, fname, old, new],
                BenchId(String("bench_string_replace", suffix)),
            )
            m.bench_function(
                bench_string_count_codepoints[length, fname],
                BenchId(String("bench_string_count_codepoints", suffix)),
            )
            m.bench_function(
                bench_string_find_single[length, fname],
                BenchId(String("bench_string_find_single", suffix)),
            )
            m.bench_function(
                bench_string_find_multiple[length, fname],
                BenchId(String("bench_string_find_multiple", suffix)),
            )
            m.bench_function(
                bench_string_is_valid_utf8[length, fname],
                BenchId(String("bench_string_is_valid_utf8", suffix)),
            )
            m.bench_function(
                bench_write_utf8[length, fname],
                BenchId(String("bench_write_utf8", suffix)),
            )
            m.bench_function(
                bench_string_repr[length, fname],
                BenchId(String("bench_string_repr", suffix)),
            )

    m.bench_function(
        bench_string_join[True],
        BenchId(String("bench_string_join_short")),
    )
    m.bench_function(
        bench_string_join[False],
        BenchId(String("bench_string_join_long")),
    )

    m.bench_function(
        bench_string_split_dense[10],
        BenchId(String("bench_string_split_dense[10]"),
    )
    )
    m.bench_function(
        bench_string_split_dense[100],
        BenchId(String("bench_string_split_dense[100]"),
    )
    )
    m.bench_function(
        bench_string_split_dense[1000],
        BenchId(String("bench_string_split_dense[1000]"),
    )
    )
    m.bench_function(
        bench_string_split_dense[10000],
        BenchId(String("bench_string_split_dense[10000]"),
    )
    )

    # NOTE: do not delete this. This is supposed to measure the average for
    # different languages. You can use print(m) if you wish to see the
    # per-language breakdown
    var results = Dict[String, Tuple[Float64, Int]]()
    for info in m.info_vec:
        var n = info.name
        var time = info.result.mean("ms")
        var avg, amnt = results.get(n, (Float64(0), 0))
        results[n] = (
            (avg * Float64(amnt) + time) / Float64((amnt + 1)),
            amnt + 1,
        )
    print("")
    for k_v in results.items():
        print(k_v.key, k_v.value[0], sep=", ")
