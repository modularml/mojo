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

from std.sys.intrinsics import prefetch

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from std.testing import assert_true


def do_prefetch[
    dtype: DType, *, offset: Int = 0
](addr: ImmPointer[Scalar[dtype], ImmutAnyOrigin]):
    prefetch(addr + offset)


def test_prefetch_nvidia() raises:
    assert_true(
        "prefetch.global.L2 "
        in _compile_code[
            do_prefetch[.float16], target=get_gpu_target["sm_80"]()
        ]()
    )
    assert_true(
        "prefetch.global.L2 "
        in _compile_code[
            do_prefetch[.float32], target=get_gpu_target["sm_80"]()
        ]()
    )
    assert_true(
        "prefetch.global.L2 "
        in _compile_code[
            do_prefetch[.int32], target=get_gpu_target["sm_80"]()
        ]()
    )

    assert_true(
        "prefetch.global.L2 "
        in _compile_code[
            do_prefetch[.int64, offset=42],
            target=get_gpu_target["sm_80"](),
        ]()
    )


def main() raises:
    test_prefetch_nvidia()
