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

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics


def takeIt[T: def () -> None, //](state: T):
    state()

struct MoveMe(Movable):
    var x:Int
    def __init__(out self, *, deinit move: Self):
        self.x = move.x
    def __deinit__(deinit self:Self):
        pass

def use(d:MoveMe):
    pass

# CHECK-LABEL:  lit.fn @"toy
def toy(var byMove: MoveMe): # expected-note {{'byMove' declared here}}
    def myclosure() {var byMove^}:
        use(byMove)

    use(byMove) # expected-error {{use of uninitialized value 'byMove'}}
    takeIt(myclosure)
