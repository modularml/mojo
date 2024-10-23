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
from random import random_si64, seed
from pathlib import _dir_of_current_file
from collections import Optional
from os import abort
from stdlib.collections.string import String
from stdlib.utils._utf8_validation import _is_valid_utf8


# ===----------------------------------------------------------------------===#
# Benchmark Data
# ===----------------------------------------------------------------------===#
fn make_string[
    length: UInt = 0
](filename: StringLiteral = "UN_charter_EN.txt") -> String:
    """Make a `String` made of items in the `./data` directory.

    Parameters:
        length: The length in bytes of the resulting `String`. If == 0 -> the
            whole file content.

    Args:
        filename: The name of the file inside the `./data` directory.
    """

    try:
        directory = _dir_of_current_file() / "data"
        var f = open(directory / filename, "rb")

        @parameter
        if length > 0:
            var items = f.read_bytes(length)
            i = 0
            while length > len(items):
                items.append(items[i])
                i = i + 1 if i < len(items) - 1 else 0
            items.append(0)
            return String(items^)
        else:
            return String(f.read_bytes())
    except e:
        print(e, file=2)
    return abort[String]()


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
    length: UInt = 0,
    filename: StringLiteral = "UN_charter_EN",
    sequence: StringLiteral = "a",
](inout b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

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
    filename: StringLiteral = "UN_charter_EN",
    sequence: Optional[StringLiteral] = None,
](inout b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

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
    length: UInt = 0, filename: StringLiteral = "UN_charter_EN"
](inout b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

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
    length: UInt = 0, filename: StringLiteral = "UN_charter_EN"
](inout b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

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
    length: UInt = 0, filename: StringLiteral = "UN_charter_EN"
](inout b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

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
    filename: StringLiteral = "UN_charter_EN",
    old: StringLiteral = "a",
    new: StringLiteral = "A",
](inout b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

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
    length: UInt = 0, filename: StringLiteral = "UN_charter_EN"
](inout b: Bencher) raises:
    var items = make_string[length](filename + ".html")

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
    var m = Bench(BenchConfig(num_repetitions=5))
    # NOTE: A proper way to run a benchmark like this is:
    # 1. Run the benchmark on nightly branch with num_repetitions=5 and take the
    #    **median** value for each function, length, and language that is to be
    #    measured.
    # 2. Then run the benchmark on num_repetitions=1 if you want faster results
    #    during development of your branch.
    # 3. When ready to make statements about speed improvements, first run the
    #    benchmark again with num_repetitions=5 and take the **median** of that.
    # 4. Make a table and report the new **median** numbers and the markdown
    #    percentage improvement over nightly version (new - nightly)/nightly.
    alias filenames = (
        "UN_charter_EN",
        "UN_charter_ES",
        "UN_charter_AR",
        "UN_charter_RU",
        "UN_charter_zh-CN",
    )
    alias old_chars = ("a", "ó", "ل", "и", "一")
    alias new_chars = ("A", "Ó", "ل", "И", "一")
    alias lengths = (10, 30, 50, 100, 1000, 10_000, 100_000, 1_000_000)

    m.bench_function[bench_string_init](BenchId("bench_string_init"))

    @parameter
    for i in range(len(lengths)):
        alias length = lengths.get[i, Int]()

        @parameter
        for j in range(len(filenames)):
            alias fname = filenames.get[j, StringLiteral]()
            alias old = old_chars.get[j, StringLiteral]()
            alias new = new_chars.get[j, StringLiteral]()
            suffix = "[" + str(length) + "](" + fname + ")"
            m.bench_function[bench_string_count[length, fname, old]](
                BenchId("bench_string_count" + suffix)
            )
            m.bench_function[bench_string_split[length, fname, old]](
                BenchId("bench_string_split" + suffix)
            )
            m.bench_function[bench_string_split[length, fname]](
                BenchId("bench_string_split_none" + suffix)
            )
            m.bench_function[bench_string_splitlines[length, fname]](
                BenchId("bench_string_splitlines" + suffix)
            )
            m.bench_function[bench_string_lower[length, fname]](
                BenchId("bench_string_lower" + suffix)
            )
            m.bench_function[bench_string_upper[length, fname]](
                BenchId("bench_string_upper" + suffix)
            )
            m.bench_function[bench_string_replace[length, fname, old, new]](
                BenchId("bench_string_replace" + suffix)
            )
            m.bench_function[bench_string_is_valid_utf8[length, fname]](
                BenchId("bench_string_is_valid_utf8" + suffix)
            )
    m.dump_report()
