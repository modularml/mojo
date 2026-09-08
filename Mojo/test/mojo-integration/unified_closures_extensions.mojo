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
# RUN: %mojo %s 3 1 4 | FileCheck %s

from std.sys import argv

# test that traits that vary by parameter name only are bridgable


def sinkNameOnly[V: Movable & Deinitable, //, F: def() -> V](*, call: F) -> V:
    return call()


def forwardNameOnly[
    T: Movable & Deinitable, //, G: def() -> T
](*, call: G) -> T:
    return sinkNameOnly(call=call)


def topNameOnly() -> Int:
    return 42


def testExtensionNameOnly(x: Int, y: Int):
    def fat() {var} -> Int:
        return x + y

    print(forwardNameOnly(call=fat))
    print(forwardNameOnly(call=topNameOnly))


# test that a trait that is more concrete but compatible is bridgable
def sinkConcrete[
    S: Movable & Deinitable,
    V: Movable & Deinitable,
    //,
    F: def(arg: V) -> S,
](*, call: F, arg: V) -> S:
    return call(arg)


def forwardConcrete[
    T: Movable & Deinitable, //, G: def(arg: T) -> Int
](*, call: G, arg: T) -> Int:
    return sinkConcrete(call=call, arg=arg)


def top(arg: Int) -> Int:
    return 42 + arg


def testExtensionConcrete(x: Int, y: Int, arg: Int):
    def fat(arg: Int) {var} -> Int:
        return x + y + arg

    print(forwardConcrete(call=fat, arg=arg))
    print(forwardConcrete(call=top, arg=arg))


# Test Structs
struct Inner[K: Movable & Deinitable]:
    def __init__(out self):
        pass

    def take(self, f: Some[def(var Self.K)], var arg: Self.K, /):
        f(arg^)


struct Outer:
    comptime key_type = String
    var _inner: Inner[Self.key_type]

    def __init__(out self):
        self._inner = Inner[Self.key_type]()

    def forward(self, f: Some[def(var String)], var arg: String, /):
        self._inner.take(f, arg^)


def testExtensionAssocType(prefix: String):
    def show(var s: String) {var}:
        print(prefix, s)

    var o = Outer()
    o.forward(show, String("world"))


def main() raises:
    var x = atol(argv()[1])
    var y = atol(argv()[2])
    var z = atol(argv()[3])
    # CHECK: 4
    # CHECK: 42
    testExtensionNameOnly(x, y)
    # CHECK: 8
    # CHECK: 46
    testExtensionConcrete(x, y, z)
    # CHECK: hello world
    testExtensionAssocType(String("hello"))
