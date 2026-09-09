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

# RUN: %parse-mojo-isolated %s -verify-diagnostics


struct WeirdArray(Movable where False):
    # expected-note @+1 {{function declared here}}
    def __getitem__(self, x: Int) -> Int:
        return x


def test_getitem(var a: WeirdArray, f: String, x: Int):
    # expected-error @+1 {{invalid call to '__getitem__': value passed to 'x' cannot be converted from 'String' to 'Int'}}
    _ = a[f]

    # expected-error @+1 {{invalid call to '__getitem__': unexpected argument}}
    _ = a[x, x]

    # expected-error @+1 {{expression must be mutable in assignment}}
    a[x] = x


struct NotSettable(Movable where False):
    def __getitem__(self) -> Int:
        pass

    # expected-error @+1 {{__setitem__ must take at least one argument for the value to set}}
    def __setitem__(self):
        pass


def test_setitem_kwargs(ns: NotSettable, x: Int):
    # expected-note @+1 {{used in an expression here}}
    ns[] = x

@fieldwise_init
struct VariadicIndexList(Movable where False):
    def __getitem__(mut self, *indices: Int) -> Int:
        pass

    def __setitem__(mut self, *indices: Int, val: Int):
        pass


# CHECK-LABEL: lit.fn @"testVariadicIndexList
# MOCO-696: Support variadic length keys in __setitem__
def testVariadicIndexList(mut foo: VariadicIndexList, i: Int, the_value: Int):
    # Getter is straight-forward.
    _ = foo[i, i]

    # Setter needs to pass the new value as 'val', not in the variadics.
    foo[i, i, i, i] = the_value
