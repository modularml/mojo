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

from std.memory import OwnedPointer


def main():
    # start-owned-pointer-create
    var ptr: OwnedPointer[Int]
    ptr = OwnedPointer(100)
    # end-owned-pointer-create

    # start-owned-pointer-dereference
    # Update an initialized value
    ptr[] += 10
    # Access an initialized value
    print(ptr[])
    # end-owned-pointer-dereference
