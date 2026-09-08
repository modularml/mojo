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
# Verifies that a negative index triggers an out of bounds assertion error,
# and reports the correct location where the negative index was provided.
#
# ===----------------------------------------------------------------------=== #

from std.collections import LinkedList


# CHECK-LABEL: test_negative_index_linked_list
def main():
    print("== test_negative_index_linked_list")
    var l = LinkedList[Int]()
    l.append(1)
    l.append(2)
    l.append(3)
    # CHECK: test_negative_index_linked_list.mojo:30:18: Assert Error: index -1 is out of bounds, valid range is 0 to 2
    _ = l.get_nth(-1)
    # CHECK-NOT: is never reached
    print("is never reached")
