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

# RUN: %parse-mojo-isolated %s | FileCheck %s


struct BoxedInt(Copyable, RegisterPassable):
    var value: Int

    @implicit
    def __init__(out self, value: Int):
        self.value = value

    def boxedAdd(self, rhs: Int) -> Int:
        return self.value + rhs


struct Param[T: TrivialRegisterPassable](Movable where False):
    pass


# CHECK-LABEL: lit.fn @"capturing_in_struct
# CHECK-SAME: capturing -> !kgen.none
def capturing_in_struct[x: Param[def() capturing -> Int]]():
    pass


# CHECK-LABEL: lit.struct.decl @CapturingMember
struct CapturingMember[f: def() capturing -> None](Movable where False):
    # CHECK-LABEL: lit.fn @"member
    # CHECK-SAME: capturing -> !kgen.none attributes
    def member(self):
        pass

    # CHECK-LABEL: lit.fn @"static_method
    # CHECK-SAME: capturing -> !kgen.none attributes
    @staticmethod
    def static_method():
        pass


def makeClosure[p: Int](x: Int) -> Int:
    var z = x + x

    # CHECK: [[COPY_VAL:%.*]] = lit.ref.load %z : <!Int, mut *"z`">
    # CHECK:  = kgen.param.constant: !Int = <p>
    @__copy_capture(z, p)
    @__parameter
    def writer() -> Int:
        # CHECK: [[REBOUND:%.*]] = kgen.rebind [[COPY_VAL]] : !Int to !alias_Int1
        # CHECK: lit.return [[REBOUND]] : !alias_Int1
        return z

    return writer()


def foo():
    var x = 3
    var y = 2
    _ = makeClosure[3](x)
