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
"""Microbenchmarks for t-string interpolation and `Writable` dispatch cost,
written through `DispatchSink`, a null writer, to isolate that cost from
real I/O.

Interpolated values are `black_box`-ed at the call site so the compiler
can't fold the formatting away as a compile-time constant.
"""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep


@fieldwise_init
struct DispatchSink(Writer):
    """`Writer` that checksums every byte instead of discarding it. `keep()`
    on a bare `StringSlice` only preserves the `{ptr, len}` descriptor, not
    the bytes, so the compiler can skip producing them.
    """

    var checksum: UInt64

    @always_inline
    def write_string(mut self, string: StringSlice):
        for byte in string.bytes():
            self.checksum = self.checksum * 31 + UInt64(byte)


@fieldwise_init
struct NullWritable(Writable):
    var string: StaticString

    @always_inline
    def write_to(self, mut writer: Some[Writer]):
        writer.write_string(self.string)


@always_inline
def dispatch_print(tstring: Some[Writable]):
    var writer = DispatchSink(0)
    tstring.write_to(writer)
    keep(writer.checksum)


def bench_tstring_single_value_dispatch(mut b: Bencher) raises:
    @always_inline
    def call_fn():
        var a = NullWritable("The quick brown fox")
        dispatch_print(t"{black_box(a)}")

    b.iter(call_fn)


def bench_tstring_only_literal_dispatch(mut b: Bencher) raises:
    @always_inline
    def call_fn():
        dispatch_print(t"The quick brown fox jumps over the lazy dog")

    b.iter(call_fn)


def bench_tstring_many_values_no_literals_dispatch(mut b: Bencher) raises:
    @always_inline
    def call_fn():
        var a = NullWritable("abcdef")
        var b = NullWritable("ghijklm")
        var c = NullWritable("nopqrstuvwxyz")
        var d = NullWritable("123")
        var e = NullWritable("4567890")
        dispatch_print(
            t"{black_box(a)}{black_box(b)}{black_box(c)}{black_box(d)}{black_box(e)}"
        )

    b.iter(call_fn)


def bench_tstring_long_literals_dispatch(mut b: Bencher) raises:
    @always_inline
    def call_fn():
        var a = NullWritable("I'm short")
        var b = NullWritable("I'm on the other hand a very long string")
        # fmt: off
        dispatch_print(
            t"Lorem ipsum dolor sit amet, consectetur adipiscing elit. {black_box(a)}"
            t" Sed do eiusmod tempor incididunt ut labore et dolore magna"
            t" aliqua. Ut enim ad minim veniam, quis nostrud {black_box(b)}"
            t" exercitation ullamco laboris nisi ut aliquip."
        )
        # fmt: on

    b.iter(call_fn)


def bench_tstring_many_values_many_literals_dispatch(mut b: Bencher) raises:
    @always_inline
    def call_fn():
        var a = NullWritable("abcdef")
        var b = NullWritable("ghijklm")
        var c = NullWritable("nopqrstuvwxyz")
        var d = NullWritable("123")
        var e = NullWritable("4567890")
        var f = NullWritable("🔥")
        var g = NullWritable("🔥🔥🔥🔥🔥🔥🔥🔥")
        var h = NullWritable("🐮")
        dispatch_print(
            t"a={black_box(a)} b={black_box(b)} c={black_box(c)}"
            t" d={black_box(d)} e={black_box(e)} f={black_box(f)}"
            t" g={black_box(g)} h={black_box(h)}"
        )

    b.iter(call_fn)


def bench_tstring_mixed_sizes_dispatch(mut b: Bencher) raises:
    @always_inline
    def call_fn():
        var id = NullWritable("7")
        var status = NullWritable("ok")
        var user = NullWritable("🔥")
        var message = NullWritable(
            "a moderately long status message describing what happened"
        )
        var trace = NullWritable(
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do"
            " eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut"
            " enim ad minim veniam, quis nostrud exercitation ullamco laboris"
            " nisi ut aliquip ex ea commodo consequat."
        )
        # fmt: off
        dispatch_print(
            t"[{black_box(id)}] Lorem ipsum dolor sit amet, consectetur"
            t" adipiscing elit, sed do eiusmod tempor incididunt ut labore et"
            t" dolore magna aliqua user={black_box(user)} status={black_box(status)}"
            t" Ut enim ad minim veniam, quis nostrud exercitation ullamco"
            t" laboris nisi ut aliquip ex ea commodo consequat"
            t" message={black_box(message)} trace={black_box(trace)} done."
        )
        # fmt: on

    b.iter(call_fn)


def main() raises:
    var m = Bench(BenchConfig(min_runtime_secs=0.01, num_repetitions=5))
    m.bench_function(
        bench_tstring_single_value_dispatch,
        BenchId("bench_tstring_single_value_dispatch"),
    )
    m.bench_function(
        bench_tstring_only_literal_dispatch,
        BenchId("bench_tstring_only_literal_dispatch"),
    )
    m.bench_function(
        bench_tstring_many_values_no_literals_dispatch,
        BenchId("bench_tstring_many_values_no_literals_dispatch"),
    )
    m.bench_function(
        bench_tstring_long_literals_dispatch,
        BenchId("bench_tstring_long_literals_dispatch"),
    )
    m.bench_function(
        bench_tstring_many_values_many_literals_dispatch,
        BenchId("bench_tstring_many_values_many_literals_dispatch"),
    )
    m.bench_function(
        bench_tstring_mixed_sizes_dispatch,
        BenchId("bench_tstring_mixed_sizes_dispatch"),
    )
    print(m)
