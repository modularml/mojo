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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# Test that the default Equatable implementation produces a clear error message
# when a field does not implement Equatable.


@fieldwise_init
struct NotEquatable(ImplicitlyCopyable):
    var x: Int


@fieldwise_init
struct HasBadField(Equatable):
    var field: NotEquatable


# CHECK: constraint failed: Could not derive Equatable for HasBadField - member field `field: NotEquatable` does not implement Equatable
def main() raises:
    var a = HasBadField(NotEquatable(1))
    var b = HasBadField(NotEquatable(1))
    _ = a == b
