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

from std.collections.string import StaticString


struct CallbackHolder:
    var callback: def(OpaquePointer[MutAnyOrigin], StaticString) thin -> None
    var len: Int

    def __init__(
        out self,
        func: def(OpaquePointer[MutAnyOrigin], StaticString) thin -> None,
    ):
        self.callback = func
        self.len = 1  # breakpoint


def main():
    def foo(x: OpaquePointer[MutAnyOrigin], y: StaticString) -> None:
        pass

    var holder = CallbackHolder(foo)
    print(holder.len)
