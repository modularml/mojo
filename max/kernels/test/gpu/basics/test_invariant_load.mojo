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

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from max.gpu.intrinsics import ldg
from layout import Layout, LayoutTensor
from std.testing import assert_true


def ldg_kernel(i8: MutPointer[Int8, MutAnyOrigin]):
    i8.store(1, ldg(i8))


def layout_kernel(
    a: LayoutTensor[mut=False, .int8, Layout.row_major(1), _],
    mut b: type_of(a[0]),
):
    b = a[0]


def test_ldg_kernel[emission_kind: StaticString]() raises -> String:
    return _compile_code[
        ldg_kernel,
        emission_kind=emission_kind,
        target=get_gpu_target["sm_90a"](),
    ]().asm


def test_ldg_kernel() raises:
    var llvm = test_ldg_kernel["llvm"]()
    assert_true("!invariant.load !1" in llvm)
    var asm = test_ldg_kernel["asm"]()
    assert_true("ld.global.nc.b8" in asm)


def test_layout_kernel[emission_kind: StaticString]() raises -> String:
    return _compile_code[
        layout_kernel,
        emission_kind=emission_kind,
        target=get_gpu_target["sm_90a"](),
    ]().asm


def test_layout_kernel() raises:
    var llvm = test_layout_kernel["llvm"]()
    assert_true("!invariant.load !1" in llvm)
    var asm = test_layout_kernel["asm"]()
    assert_true("ld.global.nc.b8" in asm)


def main() raises:
    test_ldg_kernel()
    test_layout_kernel()
