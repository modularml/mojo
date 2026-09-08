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

# Test that the default Hashable implementation produces a clear error message
# when a field does not implement Hashable.


@fieldwise_init
struct NotHashable(ImplicitlyCopyable):
    var x: Int


@fieldwise_init
struct HasBadField(Hashable):
    var field: NotHashable


# CHECK: constraint failed: Could not derive Hashable for HasBadField - member field `field: NotHashable` does not implement Hashable
def main() raises:
    var a = HasBadField(NotHashable(1))
    _ = hash(a)
