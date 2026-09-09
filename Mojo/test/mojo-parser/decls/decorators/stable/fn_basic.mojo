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

# Test that @stable decorator is recognized on functions and methods.


# CHECK: lit.fn @"stable_function()"
# CHECK-SAME: hasStableDecorator
@stable
def stable_function():
    pass


# CHECK: lit.fn @"unstable_function()"
# CHECK-NOT: hasStableDecorator
# CHECK-SAME: sourceName
def unstable_function():
    pass


# The struct must be @stable to allow @stable members inside it.
# CHECK: lit.struct.decl @TestStruct
# CHECK-SAME: hasStableDecorator
@stable
struct TestStruct(Movable where False):
    # CHECK: lit.fn @"stable_method{{.*}}TestStruct)"
    # CHECK-SAME: hasStableDecorator
    @stable
    def stable_method(self):
        pass

    # CHECK: lit.fn @"unstable_method{{.*}}TestStruct)"
    # CHECK-NOT: hasStableDecorator
    # CHECK-SAME: sourceName
    def unstable_method(self):
        pass

    # CHECK: lit.fn @"stable_static()"
    # CHECK-SAME: hasStableDecorator
    @stable
    @staticmethod
    def stable_static():
        pass
