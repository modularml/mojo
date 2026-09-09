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

# RUN: %parse-mojo-isolated -verify-diagnostics %s


# expected-note @+1 {{function declared here}}
def bind_fat_to_thin_target[g: def(y: Int) thin -> Int](x: Int):
    pass


def bind_fat_to_thin_main():
    var x = 4

    @__copy_capture(x)
    @__parameter
    def g(y: Int) -> Int:
        return x

    # expected-error @below {{'bind_fat_to_thin_target' parameter 'g' has 'def(y: Int) thin -> Int' type, but value has type 'def(y: Int) capturing thin -> Int'}}
    comptime Bound = bind_fat_to_thin_target[g]
    Bound(3)


def makeClosure(x: Int):
    var z = x + x

    @__copy_capture(z)
    @__parameter
    def writer() -> Int:
        # expected-error @below {{expression must be mutable in assignment}}
        z = z + z
        return z

    var y = writer()


@fieldwise_init
struct MemType(Movable where False):
    var a: Int

    def foo(self) -> MemType:
        return MemType(self.a + self.a)


struct NoCopyType(RegisterPassable):
    var a: Int

    @implicit
    def __init__(out self, aa: Int):
        self.a = aa

    def foo(self) -> NoCopyType:
        return NoCopyType(self.a + self.a)


@no_inline
def makeClosure(x: MemType):
    var rp: NoCopyType = NoCopyType(x.a)

    # expected-error @below {{value of type 'NoCopyType' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'}}
    # expected-note @below {{consider transferring the value with '^'}}
    @__copy_capture(rp)
    @__parameter
    def writer() -> Int:
        pass


def bad_capture(x: Int):
    var z = x

    # expected-error @below {{cannot capture unknown value 'not_a_thing'}}
    @__copy_capture(not_a_thing)
    @__parameter
    async def closure_1():
        pass

    # expected-error @below {{cannot capture unknown value 'not_a_thing'}}
    @__move_capture(not_a_thing)
    @__parameter
    async def closure_2():
        pass
