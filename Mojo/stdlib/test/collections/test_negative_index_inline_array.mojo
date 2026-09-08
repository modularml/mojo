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
#
# Verifies that negative indexing on Array triggers a deprecation assert.
#
# ===----------------------------------------------------------------------=== #


# CHECK-LABEL: test_negative_index_inline_array
def main():
    print("== test_negative_index_inline_array")
    var arr: Array[Int, 3] = [1, 2, 3]
    var i = -1
    # CHECK: test_negative_index_inline_array.mojo:25:12: Assert Error: index -1 is out of bounds, valid range is 0 to 2
    _ = arr[i]
    # CHECK-NOT: is never reached
    print("is never reached")
