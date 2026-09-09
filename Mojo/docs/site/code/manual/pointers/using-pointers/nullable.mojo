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
from std.memory.alloc import alloc, dealloc, Layout, ThinAllocation


def test_unsafe_dangling():
    # start-unsafe-dangling
    var ptr = Pointer[Int, MutUntrackedOrigin].unsafe_dangling()
    # end-unsafe-dangling

    _ = ptr


def main():
    # start-optional-pointer
    var ptr = Optional[Pointer[Int, MutUntrackedOrigin]]()
    # end-optional-pointer

    # Optional is initially None; if block is skipped.
    if ptr:
        # ptr is not None — safe to unwrap
        var p = ptr.value()
        print(p[])

    # Assign a real value to test the non-None path. The `Optional` holds an
    # untracked raw pointer, so leak the allocation and pair the pointer back
    # with its layout to release it.
    comptime layout = Layout[Int].single()
    var x_ptr: Pointer[Int, MutUntrackedOrigin] = alloc(layout).unsafe_leak()
    x_ptr.unsafe_write(42)
    ptr = x_ptr
    if ptr:
        # ptr is not None — safe to unwrap
        var p = ptr.value()
        print(p[])
    x_ptr.unsafe_deinit_pointee()
    dealloc(ThinAllocation(unsafe_owned_ptr=x_ptr).unsafe_with_layout(layout))

    test_unsafe_dangling()
