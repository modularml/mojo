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
# RUN: %mojo-no-debug --target-accelerator=nvidia:sm_80 %s | FileCheck --check-prefix=CHECK-NV80 %s
# RUN: %mojo-no-debug --target-accelerator=nvidia:sm_90a %s | FileCheck --check-prefix=CHECK-NV90 %s
# RUN: %mojo-no-debug --target-accelerator=nvidia:sm_120a %s | FileCheck --check-prefix=CHECK-NV120 %s

from std.sys.info import _accelerator_arch, _is_sm_9x, _is_sm_9x_or_newer

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from std.testing import *


def check_sm9x() -> Bool:
    comptime v = _is_sm_9x()
    return v


def check_sm9x_or_newer() -> Bool:
    comptime v = _is_sm_9x_or_newer()
    return v


def main() raises:
    comptime accelerator_arch = _accelerator_arch()

    # CHECK-NV80: ret i1 false
    # CHECK-NV90: ret i1 true
    # CHECK-NV120: ret i1 false
    print(
        _compile_code[
            check_sm9x,
            emission_kind="llvm",
            target=get_gpu_target[_accelerator_arch()](),
        ]()
    )

    # CHECK-NV80: ret i1 false
    # CHECK-NV90: ret i1 true
    # CHECK-NV120: ret i1 true
    print(
        _compile_code[
            check_sm9x_or_newer,
            emission_kind="llvm",
            target=get_gpu_target[_accelerator_arch()](),
        ]()
    )
