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

# RUN: kgen -elaborate %s --verify-diagnostics

# Verifies that a type error inside a mutually
# recursive function (an elaboration SCC) is surfaced through the SCC boundary
# to the caller.


# expected-note @below {{function instantiation failed}}
def check_allowed[T: AnyType]():
    comptime assert (  # expected-note {{constraint failed: type not allowed}}
        False
    ), "type not allowed"


trait Visitor:
    def visit(self, x: Int) raises -> Int:
        return 0


# expected-note @below {{function instantiation failed}}
def dispatch[V: Visitor](v: V, x: Int) raises -> Int:
    # expected-note @below {{call expansion failed}}
    return v.visit(x)


struct BadVisitor(Visitor):
    def __init__(out self):
        pass

    # expected-note @below {{function instantiation failed}}
    def visit(self, x: Int) raises -> Int:
        # dispatch(self, x) is the intra-SCC back-edge and is not in the chain.
        _ = dispatch(self, x)
        # expected-note @below {{call expansion failed}}
        check_allowed[String]()
        return 0


# expected-error @below {{function instantiation failed}}
def main() raises:
    var v = BadVisitor()
    # expected-note @below {{call expansion failed with parameter value(s): (...)}}
    _ = dispatch(v, 1)
