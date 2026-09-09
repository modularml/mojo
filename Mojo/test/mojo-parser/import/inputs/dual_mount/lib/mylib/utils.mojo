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

# Reachable as `utils` with `lib/mylib` as an import root, and as `mylib.utils`
# with `lib`. Used by import_dual_mount_module.mojo.


struct Thing:
    var x: Int

    def __init__(out self):
        self.x = 7


def make() -> Thing:
    return Thing()
