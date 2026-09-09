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
from std.testing import assert_equal, TestSuite


def _idx[target: __mlir_type.`!kgen.target`]() -> Int:
    return Int(SIMDLength(mlir_value=STD_PLUGINS.index_for_target[target]))


# STD_PLUGINS order: [DefaultPlugin=0, MetalPlugin=1, CUDAPlugin=2, HIPPlugin=3]

comptime _t_default = __mlir_attr[
    `#kgen.target<triple = "x86_64-unknown-linux-gnu", arch = "x86-64",`,
    `stdlib_plugin = "default"> : !kgen.target`,
]
comptime _t_metal = __mlir_attr[
    `#kgen.target<triple = "x86_64-unknown-linux-gnu", arch = "x86-64",`,
    `stdlib_plugin = "metal"> : !kgen.target`,
]
comptime _t_cuda = __mlir_attr[
    `#kgen.target<triple = "x86_64-unknown-linux-gnu", arch = "x86-64",`,
    `stdlib_plugin = "cuda"> : !kgen.target`,
]
comptime _t_hip = __mlir_attr[
    `#kgen.target<triple = "x86_64-unknown-linux-gnu", arch = "x86-64",`,
    `stdlib_plugin = "hip"> : !kgen.target`,
]
comptime _t_unset = __mlir_attr[
    `#kgen.target<triple = "x86_64-unknown-linux-gnu", arch = "x86-64"> : !kgen.target`,
]


def test_explicit_plugin_selection() raises:
    assert_equal(_idx[_t_default](), 0)
    assert_equal(_idx[_t_metal](), 1)
    assert_equal(_idx[_t_cuda](), 2)
    assert_equal(_idx[_t_hip](), 3)


def test_unset_defaults_to_default_plugin() raises:
    # stdlib_plugin defaults to "default" when omitted -> DefaultPlugin (idx 0).
    assert_equal(_idx[_t_unset](), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
