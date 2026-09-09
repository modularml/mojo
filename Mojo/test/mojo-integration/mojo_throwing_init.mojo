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
# RUN: %mojo -debug-level full %s | FileCheck %s


struct MemType1(ImplicitlyCopyable):
    var value: Int

    @implicit
    def __init__(out self, v: Int):
        self.value = v

    def __init__(out self, *, copy: Self):
        self.value = copy.value + 1
        print("Copy to", self.value)

    def __deinit__(deinit self):
        print("MemType1(", self.value, ") destroyed")


struct PartialInitType:
    var mem1: MemType1
    var mem2: MemType1
    var mem3: MemType1
    var setTwice: MemType1

    def __init__(out self, cond: Int, other: MemType1) raises:
        self.mem1 = other
        self.mem2 = MemType1(2)

        # This copy is entirely elided since it is dead.
        self.setTwice = other
        if cond > 2:
            raise Error("bail on init")
        self.setTwice = MemType1(98)
        self.mem3 = MemType1(3)

    def __deinit__(deinit self):
        print("destroy PartialInitType")


def main():
    print("start")
    # CHECK: start
    # CHECK-NOT: destroy PartialInitType
    # CHECK-NEXT: Copy to 43
    # CHECK-NEXT: MemType1( 43 ) destroyed
    # CHECK-NEXT: MemType1( 2 ) destroyed
    # CHECK-NEXT: MemType1( 42 ) destroyed
    # CHECK-NOT: MemType1( 3 ) destroyed
    # CHECK-NEXT: bail on init
    # CHECK-NEXT: done
    try:
        var m = MemType1(42)
        var x = PartialInitType(3, m)
    except e:
        print(e)

    print("done")
