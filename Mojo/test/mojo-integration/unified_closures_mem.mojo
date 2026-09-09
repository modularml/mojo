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
# RUN: %mojo %s AA BB CC foo | FileCheck %s


# COM: Check that the argument is augmented at the definition site.

from std.sys import argv


struct Mem(ImplicitlyCopyable):
    var str1: String
    var str2: String

    def __init__(out self, str1: String, str2: String):
        self.str1 = str1
        self.str2 = str2

    def to_string(self) -> String:
        return self.str1 + self.str2


struct MovableMem(ImplicitlyCopyable):
    var str1: String
    var str2: String

    def __init__(out self, str1: String, str2: String):
        self.str1 = str1
        self.str2 = str2

    def __init__(out self, *, copy: Self):
        self.str1 = copy.str1
        self.str2 = copy.str2
        print("copied mem")

    def to_string(self) -> String:
        return self.str1 + self.str2


def takeIt[T: def() -> String](state: T):
    print("captures: ", state())


def main() raises:
    var byCopy: String = String(unsafe_from_utf8=argv()[1].as_bytes())

    def mutateCopy() {var byCopy} -> String:
        byCopy += ".v2"
        return byCopy

    # CHECK: captures: AA.v2
    takeIt(mutateCopy)
    # CHECK: AA
    print(byCopy)

    var byMutRef: String = String(unsafe_from_utf8=argv()[2].as_bytes())

    def mutateRef() {mut byMutRef} -> String:
        byMutRef += ".v2"
        return byMutRef

    # CHECK: captures: BB.v2
    takeIt(mutateRef)
    # CHECK: BB.v2
    print(byMutRef)

    var byRef: String = String(unsafe_from_utf8=argv()[3].as_bytes())

    def immRef() {imm byRef} -> String:
        return byRef

    # CHECK: captures: CC
    takeIt(immRef)
    byRef += ".v2"

    # CHECK: CC.v2
    takeIt(immRef)

    # COM: nonmovable types can be captured by copy.
    var x: String = argv()[4]
    var mem = Mem(x, x)

    def myclosure() {var} -> String:
        return mem.to_string()

    # CHECK: captures:  foofoo
    takeIt(myclosure)
    var movableMem = MovableMem(x, x)

    # COM: Copyable closures
    @no_inline
    def copyMem() {var^}:
        print(movableMem.to_string())

    # CHECK: copied mem
    var copyOfClosure = copyMem
    # CHECK: foofoo
    copyMem()
    # CHECK: foofoo
    copyOfClosure()
