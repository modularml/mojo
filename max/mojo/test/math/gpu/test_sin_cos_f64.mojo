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
# This file only tests that `sin` and `cos` on float64 are unsupported.
#
# ===----------------------------------------------------------------------=== #
# RUN: not %bare-mojo %s 2>&1 | FileCheck %s

from std.compile import compile_info
from std.math.math import sin, cos
from max.gpu.host.info import _get_h100_target


def sin_func(x: Float64) raises -> Float64:
    return sin(x)


def cos_func(x: Float64) raises -> Float64:
    return cos(x)


def main() raises:
    print(
        compile_info[
            sin_func, emission_kind="llvm", target=_get_h100_target()
        ]()
    )
    print(
        compile_info[
            cos_func, emission_kind="llvm", target=_get_h100_target()
        ]()
    )


# Offload diagnostics are emitted in bundling order, which is sorted by
# mangled kernel name, hence cos before sin despite source order.
# CHECK: constraint failed: DType.float64 is not supported for cos on NVIDIA GPU
# CHECK: constraint failed: DType.float64 is not supported for sin on NVIDIA GPU
