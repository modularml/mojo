# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo-no-debug %s -t

from benchmark import Bench, BenchConfig, Bencher, BenchId, Unit, keep, run
from random import random_si64
from pathlib import cwd
from collections import Optional
from utils._utf8_validation import _is_valid_utf8


# ===----------------------------------------------------------------------===#
# Benchmark Data
# ===----------------------------------------------------------------------===#
fn make_string[
    length: UInt = 0, filename: String = "UN charter EN.txt"
]() -> String:
    """Make a `String` made of items in the `./data` directory or random bytes
    (ASCII value range) in case opening the file fails.

    Parameters:
        length: The length in bytes of the resulting `String`. If == 0 -> the
            whole file content.
        filename: The name of the file inside the `./data` directory.
    """

    try:
        var f = open(cwd() / "data" / filename, "rb")

        @parameter
        if length > 0:
            var items = f.read_bytes(length)
            for i in range(length - len(items)):
                items.append(items[i])
            items.append(0)
            return String(items^)
        else:
            return String(f.read_bytes())
    except:
        print("open file failed, reverting to random bytes")
        var items = List[UInt8, hint_trivial_type=True](capacity=length + 1)
        for i in range(length):
            items[i] = random_si64(0, 0b0111_1111).cast[DType.uint8]()
        items[length] = 0
        return String(items^)


# ===----------------------------------------------------------------------===#
# Benchmark string init
# ===----------------------------------------------------------------------===#
@parameter
fn bench_string_init(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn():
        for _ in range(1000):
            var d = String()
            keep(d._buffer.data)

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark string count
# ===----------------------------------------------------------------------===#
@parameter
fn bench_string_count[
    length: UInt = 0, filename: String = "UN charter EN", sequence: String = "a"
](inout b: Bencher) raises:
    var items = make_string[length, filename + ".txt"]()

    @always_inline
    @parameter
    fn call_fn() raises:
        var amnt = items.count(sequence)
        keep(amnt)

    b.iter[call_fn]()
    keep(bool(items))


# ===----------------------------------------------------------------------===#
# Benchmark string split
# ===----------------------------------------------------------------------===#
@parameter
fn bench_string_split[
    length: UInt = 0,
    filename: String = "UN charter EN",
    sequence: Optional[String] = None,
](inout b: Bencher) raises:
    var items = make_string[length, filename + ".txt"]()

    @always_inline
    @parameter
    fn call_fn() raises:
        var res: List[String]

        @parameter
        if sequence:
            res = items.split(sequence.value())
        else:
            res = items.split()
        keep(res.data)

    b.iter[call_fn]()
    keep(bool(items))


# ===----------------------------------------------------------------------===#
# Benchmark string splitlines
# ===----------------------------------------------------------------------===#
@parameter
fn bench_string_splitlines[
    length: UInt = 0, filename: String = "UN charter EN"
](inout b: Bencher) raises:
    var items = make_string[length, filename + ".txt"]()

    @always_inline
    @parameter
    fn call_fn() raises:
        var res = items.splitlines()
        keep(res.data)

    b.iter[call_fn]()
    keep(bool(items))


# ===----------------------------------------------------------------------===#
# Benchmark string lower
# ===----------------------------------------------------------------------===#
@parameter
fn bench_string_lower[
    length: UInt = 0, filename: String = "UN charter EN"
](inout b: Bencher) raises:
    var items = make_string[length, filename + ".txt"]()

    @always_inline
    @parameter
    fn call_fn() raises:
        var res = items.lower()
        keep(res._buffer.data)

    b.iter[call_fn]()
    keep(bool(items))


# ===----------------------------------------------------------------------===#
# Benchmark string upper
# ===----------------------------------------------------------------------===#
@parameter
fn bench_string_upper[
    length: UInt = 0, filename: String = "UN charter EN"
](inout b: Bencher) raises:
    var items = make_string[length, filename + ".txt"]()

    @always_inline
    @parameter
    fn call_fn() raises:
        var res = items.upper()
        keep(res._buffer.data)

    b.iter[call_fn]()
    keep(bool(items))


# ===----------------------------------------------------------------------===#
# Benchmark string replace
# ===----------------------------------------------------------------------===#
@parameter
fn bench_string_replace[
    length: UInt = 0,
    filename: String = "UN charter EN",
    old: String = "a",
    new: String = "A",
](inout b: Bencher) raises:
    var items = make_string[length, filename + ".txt"]()

    @always_inline
    @parameter
    fn call_fn() raises:
        var res = items.replace(old, new)
        keep(res._buffer.data)

    b.iter[call_fn]()
    keep(bool(items))


# ===----------------------------------------------------------------------===#
# Benchmark string _is_valid_utf8
# ===----------------------------------------------------------------------===#
@parameter
fn bench_string_is_valid_utf8[
    length: UInt = 0, filename: String = "UN charter EN"
](inout b: Bencher) raises:
    var items = make_string[length, filename + ".html"]()

    @always_inline
    @parameter
    fn call_fn() raises:
        var res = _is_valid_utf8(items.unsafe_ptr(), length)
        keep(res)

    b.iter[call_fn]()
    keep(bool(items))


# ===----------------------------------------------------------------------===#
# Benchmark Main
# ===----------------------------------------------------------------------===#
def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=1))
    m.bench_function[bench_string_init](BenchId("bench_string_init"))
    alias filenames = (
        "UN charter EN",
        "UN charter ES",
        "UN charter AR",
        "UN charter RU",
        "UN charter zh-CN",
    )
    alias old_new_chars = (
        ("a", "A"),
        ("ó", "Ó"),
        ("ل", "ل"),
        ("и", "И"),
        ("一", "一"),
    )
    alias lengths = (
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10_000,
        20_000,
        30_000,
        40_000,
        50_000,
        60_000,
        70_000,
        80_000,
        90_000,
        100_000,
        200_000,
        300_000,
        400_000,
        500_000,
        600_000,
        700_000,
        800_000,
        900_000,
        1_000_000,
    )

    @parameter
    for i in range(len(lengths)):
        alias length = lengths.get[i, Int]()

        @parameter
        for j in range(len(filenames)):
            alias fname = filenames.get[j, StringLiteral]()
            alias chars = old_new_chars.get[j, Tuple[String, String]]()
            alias old = chars.get[0, String]()
            alias new = chars.get[1, String]()
            m.bench_function[bench_string_count[length, fname, old]](
                BenchId("bench_string_count[" + str(length) + "]")
            )
            m.bench_function[bench_string_split[length, fname, old]](
                BenchId("bench_string_split[" + str(length) + "]")
            )
            m.bench_function[bench_string_split[length, fname]](
                BenchId(
                    "bench_string_split[" + str(length) + ", sequence=None]"
                )
            )
            m.bench_function[bench_string_splitlines[length, fname]](
                BenchId("bench_string_splitlines[" + str(length) + "]")
            )
            m.bench_function[bench_string_lower[length, fname]](
                BenchId("bench_string_lower[" + str(length) + "]")
            )
            m.bench_function[bench_string_upper[length, fname]](
                BenchId("bench_string_upper[" + str(length) + "]")
            )
            m.bench_function[bench_string_replace[length, fname, old, new]](
                BenchId("bench_string_replace[" + str(length) + "]")
            )
            m.bench_function[bench_string_is_valid_utf8[length, fname]](
                BenchId("bench_string_is_valid_utf8[" + str(length) + "]")
            )
    m.dump_report()
