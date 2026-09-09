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

# RUN: %mojo -debug-level full %s | FileCheck %s


def takes_int_variadic_kwargs(var **kwargs: Int) raises:
    var key = "stuff"
    # CHECK: stuff 8
    print(key, kwargs[key])
    # CHECK: x 9
    print("x", kwargs["x"])

    try:
        _ = kwargs["non-existent"]
    except:
        # CHECK: non-existent key not found (as expected)
        print("non-existent key not found (as expected)")


trait Resettable(Deinitable, ImplicitlyCopyable):
    def reset(mut self):
        ...

    def get(self) -> Int:
        ...


@fieldwise_init
struct MemOnly(Resettable):
    var value: Int

    def get(self) -> Int:
        return self.value

    def reset(mut self):
        self.value = 0


def takes_mem_only_variadic_kwargs[T: Resettable](var **kwargs: T) raises:
    var key = "fizzbuzz"
    # CHECK: fizzbuzz 13
    print(key, kwargs[key].get())
    # CHECK: y 42
    print("y", kwargs["y"].get())

    try:
        _ = kwargs[""]
    except:
        # CHECK: empty key not found
        print("empty key not found")

    kwargs[key].reset()
    # CHECK: fizzbuzz now 0
    print(key, "now", kwargs[key].get())


def main() raises:
    takes_int_variadic_kwargs(x=9, stuff=8)

    var m = MemOnly(42)
    takes_mem_only_variadic_kwargs(y=m, fizzbuzz=MemOnly(13))
    # CHECK: m outside 42
    print("m outside", m.value)

    forwards_both_variadics(1, 2, x=8, y=9)
    closure_forwards_both_variadics()
    closure_forwards_mixed()

    kitchen_sink(1, 2, 3, 4, 5, named=6, opt=7, k=8, z=9)
    forwards_to_kitchen_sink(40, 50, k=70, z=80)


def takes_both_variadics(*args: Int, var **kwargs: Int) raises:
    # Two subscripts of one dict in a single call trip interior-origin
    # exclusivity checking, so hoist them.
    var x = kwargs["x"]
    var y = kwargs["y"]
    # CHECK: forwarded 1 2 8 9
    print("forwarded", args[0], args[1], x, y)


def forwards_both_variadics(*args: Int, var **kwargs: Int) raises:
    takes_both_variadics(*args, **kwargs^)


def closure_forwards_both_variadics() raises:
    var z = 100

    def c(*args: Int, var **kwargs: Int) raises {imm z} -> Int:
        return z + args[0] + kwargs["a"]

    # CHECK: closure 108
    print("closure", c(3, a=5))


def closure_forwards_mixed() raises:
    var z = 1000

    def m(
        x: Int, *args: Int, named: Int, var **kwargs: Int
    ) raises {imm z} -> Int:
        return z + x + args[0] + named + kwargs["a"]

    # CHECK: mixed 1032
    print("mixed", m(1, 2, named=4, a=25))


# Every argument kind in one signature: positional-only ('/'), positional,
# variadic, named keyword-only (required and defaulted), and **kwargs.
def kitchen_sink(
    a: Int,
    b: Int,
    /,
    c: Int,
    *args: Int,
    named: Int,
    opt: Int = 9,
    var **kwargs: Int,
) raises:
    var k = kwargs["k"]
    var z = kwargs["z"]
    # CHECK: sink 1 2 3 4 5 6 7 8 9
    # CHECK: sink 1 2 3 40 50 6 9 70 80
    print("sink", a, b, c, args[0], args[1], named, opt, k, z)


def forwards_to_kitchen_sink(*args: Int, var **kwargs: Int) raises:
    kitchen_sink(1, 2, 3, *args, named=6, **kwargs^)
