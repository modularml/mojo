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

# RUN: %mojo -debug-level full %s 2 3 | FileCheck %s


trait A:
    pass


trait B:
    pass


trait C:
    pass


struct S(A, B):
    pass


struct S1(C):
    pass


def foo[T: AnyType]():
    comptime if conforms_to(T, A & B):
        print("T conforms to 'A & B'")
    else:
        print("T does not conform to 'A & B'")


def bar[T: AnyType]() where conforms_to(T, A & B):
    print("overload A")
    return


def bar[T: AnyType]() where conforms_to(T, C):
    print("overload B")
    return


def indirect_target[T: AnyType]() where conforms_to(T, C):
    print("selected indirectly")
    return


def indirect[T: C]():
    indirect_target[T]()
    return


def main():
    # CHECK: S conforms to 'A & B'
    comptime if conforms_to(S, A & B):
        print("S conforms to 'A & B'")

    # CHECK: S does not conform to 'C'
    comptime if not conforms_to(S, C):
        print("S does not conform to 'C'")

    # CHECK: T conforms to 'A & B'
    foo[S]()

    # CHECK: T does not conform to 'A & B'
    foo[S1]()

    # CHECK: overload A
    bar[S]()

    # CHECK: overload B
    bar[S1]()

    # CHECK: selected indirectly
    indirect[S1]()
