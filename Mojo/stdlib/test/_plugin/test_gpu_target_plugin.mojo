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

from std._plugin._overlay import STD_PLUGINS
from max.gpu.host.info import (
    _get_a100_target,
    _get_h100_target,
    _get_metal_m1_target,
    _get_mi300x_target,
)
from std.testing import assert_equal, TestSuite


# STD_PLUGINS order: [DefaultPlugin=0, MetalPlugin=1, CUDAPlugin=2, HIPPlugin=3]


def _idx[target: __mlir_type.`!kgen.target`]() -> Int:
    return Int(SIMDLength(mlir_value=STD_PLUGINS.index_for_target[target]))


def test_nvidia_targets_select_cuda_plugin() raises:
    assert_equal(_idx[_get_a100_target()](), 2)
    assert_equal(_idx[_get_h100_target()](), 2)


def test_amd_targets_select_hip_plugin() raises:
    assert_equal(_idx[_get_mi300x_target()](), 3)


def test_apple_targets_select_metal_plugin() raises:
    assert_equal(_idx[_get_metal_m1_target()](), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
