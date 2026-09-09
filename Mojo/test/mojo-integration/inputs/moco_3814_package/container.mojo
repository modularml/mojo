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

from .inner import Inner


struct Container:
    var count: Int

    def __init__(out self, count: Int):
        self.count = count

    def get_count(self) -> Int:
        return self.count

    # This method references Inner in its return type but is not called by
    # the importing test.  When Container is body-resolved, this FnOp is
    # materialized (placed in parsedDeclList as 'unparsed') with Inner
    # appearing in its function-type attribute.
    def get_inner(self) -> Inner:
        return Inner(self.count)
