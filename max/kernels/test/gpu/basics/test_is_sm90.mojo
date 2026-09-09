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

from std.sys.info import _is_sm_9x

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from std.testing import *


def check_sm() -> Bool:
    comptime v = _is_sm_9x()
    return v


def test_is_sm_9x() raises:
    assert_true(
        "ret i1 true"
        in _compile_code[
            check_sm,
            emission_kind="llvm",
            target=get_gpu_target["sm_90"](),
        ]()
    )
    assert_true(
        "ret i1 true"
        in _compile_code[
            check_sm,
            emission_kind="llvm",
            target=get_gpu_target["sm_90a"](),
        ]()
    )


def main() raises:
    test_is_sm_9x()
