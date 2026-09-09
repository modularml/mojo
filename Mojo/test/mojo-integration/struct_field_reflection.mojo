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
# RUN: %mojo %s | FileCheck %s


from std.builtin.rebind import downcast


trait CanDoSomething:
    # An unsafe default that assumes all fields implement CanDoSomething.
    def do_something(self):
        comptime names = reflect[Self].field_names()

        comptime for i in range(names.length):
            print(materialize[names[i]](), ": ", sep="", end="")
            comptime assert conforms_to(
                reflect[Self].field_types()[i], CanDoSomething
            )
            reflect[Self].field_ref[i](self).do_something()


@fieldwise_init
struct Overriding(CanDoSomething, ImplicitlyCopyable):
    def do_something(self):
        print("overriding")


@fieldwise_init
struct Overriding2(CanDoSomething, ImplicitlyCopyable):
    def do_something(self):
        print("overriding2")


@fieldwise_init
struct WrapperStruct(CanDoSomething):
    var x: Overriding
    var y: Overriding2


def call_do_something[T: CanDoSomething](ref t: T):
    t.do_something()


def closure_fields():
    var a: Int32 = 42
    var b: Int32 = 27

    def test() {var a, var b} -> Int32:
        return a + b

    # COM: reset `b` value to be 31
    __struct_field_ref(1, test) = Int32(31)

    print("closure_fields: ", test())

    # COM: reset `b` value to be 100
    __struct_field_ref(1, test) = Int32(100)

    print("closure_fields: ", test())


def main():
    var my_struct = WrapperStruct(Overriding(), Overriding2())
    call_do_something(my_struct)
    # CHECK: x: overriding
    # CHECK: y: overriding2

    closure_fields()
    # CHECK: closure_fields: 73
    # CHECK: closure_fields: 142
