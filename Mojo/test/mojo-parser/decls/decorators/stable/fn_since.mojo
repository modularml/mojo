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

# Test that @stable(since="version") stores the version string in the IR.


# CHECK: lit.fn @"fn_since_version()"
# CHECK-SAME: hasStableDecorator
# CHECK-SAME: stableSinceVersion = "1.0"
@stable(since="1.0")
def fn_since_version():
    pass


# CHECK: lit.fn @"fn_since_relaxed_semver()"
# CHECK-SAME: hasStableDecorator
# CHECK-SAME: stableSinceVersion = "2.1.3rc1"
@stable(since="2.1.3rc1")
def fn_since_relaxed_semver():
    pass


# Bare @stable must NOT produce a stableSinceVersion attribute.
# CHECK: lit.fn @"fn_bare_stable()"
# CHECK-SAME: hasStableDecorator
# CHECK-NOT: stableSinceVersion
@stable
def fn_bare_stable():
    pass


# @stable(since=) also works on structs and their members.
# CHECK: lit.struct.decl @StableStruct
# CHECK-SAME: hasStableDecorator
# CHECK-SAME: stableSinceVersion = "1.0"
@stable(since="1.0")
struct StableStruct(Movable where False):
    # CHECK: lit.fn @"method{{.*}}StableStruct)"
    # CHECK-SAME: hasStableDecorator
    # CHECK-SAME: stableSinceVersion = "1.1"
    @stable(since="1.1")
    def method(self):
        pass
