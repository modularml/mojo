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


trait Trait1:
    def f1(self):
        ...


trait Trait2:
    def f2(self):
        ...


trait Trait3:
    def f3(self):
        ...


comptime Traits12 = Trait1 & Trait2
comptime Traits23 = Trait2 & Trait3
comptime Traits123 = Trait1 & Trait2 & Trait3


@fieldwise_init
struct Struct4(Movable where False):
    def f4(self):
        pass


# expected-note @below {{function declared here}}
def use1[T: Trait1](x: T):
    pass


# Use aliased trait composition.
# expected-note @below {{function declared here}}
def use12Alias[T: Traits12](x: T):
    pass


# Use direct trait composition.
# expected-note @below {{function declared here}}
def use12Direct[T: Trait1 & Trait2](x: T):
    pass


# CHECK: lit.fn @"main_use()"
def main_use():
    var s4 = Struct4()

    # expected-error @below {{invalid call to 'use1': value passed to 'x' cannot be converted from 'Struct4' to 'T', argument type 'Struct4' does not conform to trait 'Trait1'}}
    use1(s4)

    # expected-error @below {{invalid call to 'use12Alias': value passed to 'x' cannot be converted from 'Struct4' to 'T', argument type 'Struct4' does not conform to trait 'Traits12'}}
    # expected-note @below {{'Traits12' is aka 'Trait1 & Trait2'}}
    use12Alias(s4)

    # expected-error @below {{invalid call to 'use12Direct': value passed to 'x' cannot be converted from 'Struct4' to 'T', argument type 'Struct4' does not conform to trait 'Trait1 & Trait2'}}
    use12Direct(s4)
