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

# Test that @stable decorator is recognized on structs and sets the
# hasStableDecorator attribute in the IR.


# CHECK: lit.struct.decl @StableStruct
# CHECK-SAME: hasStableDecorator
@stable
struct StableStruct(Movable where False):
    pass


# CHECK: lit.struct.decl @UnstableStruct
# CHECK-NOT: hasStableDecorator
# CHECK-SAME: sourceName
struct UnstableStruct(Movable where False):
    pass


# Verify @stable works when combined with other decorators.
# The choice of @fieldwise_init is arbitrary - any struct decorator
# works. This test ensures decorator composition doesn't break @stable.
# CHECK: lit.struct.decl @StableWithOtherDecorators
# CHECK-SAME: hasStableDecorator
@stable
@fieldwise_init
struct StableWithOtherDecorators(RegisterPassable):
    pass
