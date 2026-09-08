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

from std.math import sqrt


# start-rsqrt-def
def rsqrt[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    return 1 / sqrt(x)


# end-rsqrt-def


# start-infer-struct-param
struct One[Type: Writable & Copyable & Deinitable]:
    var value: Self.Type

    def __init__(out self, value: Self.Type):
        self.value = value.copy()


def use_one():
    var s1 = One(123)  # equivalent to One[Int](123)
    var s2 = One("Hello")  # equivalent to One[String]("Hello")
    # end-infer-struct-param
    _ = s1^
    _ = s2^


# start-infer-constructor-static-param
struct Two[Type: Writable & Copyable & Deinitable]:
    var val1: Self.Type
    var val2: Self.Type

    def __init__(out self, one: One[Self.Type], another: One[Self.Type]):
        self.val1 = one.value.copy()
        self.val2 = another.value.copy()
        print(String(self.val1), String(self.val2))

    @staticmethod
    def fire(thing1: One[Self.Type], thing2: One[Self.Type]):
        print("🔥", String(thing1.value), String(thing2.value))


def use_two():
    var s3 = Two(One("infer"), One("me"))
    Two.fire(One(1), One(2))
    # Two.fire(One("mixed"), One(0)) # Error: parameter inferred to two different values
    # end-infer-constructor-static-param
    _ = s3^


def main():
    # start-rsqrt-usage
    var v = Scalar[.float16](42)
    print(rsqrt(v))
    # end-rsqrt-usage

    # second example
    use_one()

    # third example
    use_two()
