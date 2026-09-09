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
#
# This file tests that `sqrt` on float64 lowers to the IEEE correctly-rounded
# hardware sqrt (`sqrt.rn.f64`) on NVIDIA GPU. NVIDIA has no approximate f64
# sqrt, so the generic intrinsic path is used instead.
#
# ===----------------------------------------------------------------------=== #
# RUN: %bare-mojo %s 2>&1 | FileCheck %s

from std.compile import compile_info
from std.math.math import sqrt
from max.gpu.host.info import _get_h100_target


def sqrt_func(x: Float64) raises -> Float64:
    # CHECK: sqrt.rn.f64
    return sqrt(x)


def main() raises:
    print(
        compile_info[
            sqrt_func, emission_kind="asm", target=_get_h100_target()
        ]()
    )
