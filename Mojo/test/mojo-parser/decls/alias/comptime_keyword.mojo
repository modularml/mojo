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

# RUN: %parse-mojo-isolated %s

# Test that 'comptime' keyword works as a synonym for 'alias'

# Simple comptime declarations
comptime x = 5
comptime y: Int = 10
comptime z = x + y

# Parametric comptime
comptime MyInt[T: AnyType] = T
comptime Add[a: Int, b: Int] = a + b


# In struct
struct MyStruct(Movable where False):
    comptime SIZE = 100
    comptime Type = Int

    def use_comptime(self) -> Int:
        return Self.SIZE


# In trait
trait MyTrait:
    comptime AssociatedType: AnyType


# Mixing alias and comptime in same file (both should work)
comptime old_style = 42
comptime new_style = 42
