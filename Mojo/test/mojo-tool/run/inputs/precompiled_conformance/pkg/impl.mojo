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

# Referenced only through the `default_helper` alias, so a consumer that
# imports `Container` never materializes `DefaultHelper`'s declaration.

from .helper import Helper


@fieldwise_init
struct DefaultHelper[n: Int](Copyable, Helper, Movable):
    var x: Int

    def help(self):
        pass
