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


struct PCall(Movable where False):
    def __init__(out self):
        pass

    def __call__[x: Int](ref self, y: Int) -> Int:
        return x + y


struct PCallWithGetItem(Movable where False):
    def __init__(out self):
        pass

    # expected-warning @below {{parametric '__call__' method cannot be called directly because 'PCallWithGetItem' defines '__getitem__', '__setitem__', or '__getattr__'; consider using a different name for this method}}
    def __call__[x: Int](ref self, y: Int) -> Int:
        return x + y

    # expected-note @below {{__getitem__ defined here}}
    def __getitem__(ref self, y: Int) -> Int:
        return y

    # expected-note @below {{__getitem__ defined here}}
    def __getitem__(ref self, y: StringLiteral) -> Int:
        return 2

struct PCallWithGetAttr(Movable where False):
    def __init__(out self):
        pass

    # expected-warning @below {{parametric '__call__' method cannot be called directly because 'PCallWithGetAttr' defines '__getitem__', '__setitem__', or '__getattr__'; consider using a different name for this method}}
    def __call__[_x: StringLiteral](ref self):
        pass

    # expected-note @below {{__getattr__ defined here}}
    def __getattr__[x: StringLiteral](self) -> Int:
        return 2


# Note: Test heuristic fix for MOCO-2833
struct PCallWithGetItemAndInferredParameters(Movable where False):
    def __init__(out self):
        pass

    # This does *not* produce a warning, because the parameters are all
    # inferred.
    def __call__[T: Movable, //](ref self, var y: T) -> T:
        return y^

    def __getitem__(ref self, y: Int) -> Int:
        return y


def main():
    var pc = PCall()
    _ = pc[1](2)

    # verifying that we don't break structs with __getitem__ defined
    var pcgi = PCallWithGetItem()
    _ = pcgi[1](  # expected-error {{'Int' does not implement the '__call__' method}}
        2
    )
