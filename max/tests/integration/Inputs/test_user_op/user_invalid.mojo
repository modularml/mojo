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

import extensibility
from extensibility import OutputTensor


@extensibility.register("fails_to_elaborate")
struct FailsToElaborate:
    @staticmethod
    def execute(
        output: OutputTensor[dtype=.int32, rank=1, ...],
    ):
        comptime assert False, "oops"
