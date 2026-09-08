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

from max.gpu import thread_idx, WARP_SIZE
from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from std.math.uutils import ufloordiv
from std.sys.intrinsics import readfirstlane
from std.testing import assert_true

comptime MI355X_TARGET = get_gpu_target["mi355x"]()


def readfirstlane_kernel[
    dtype: DType
](y: MutPointer[Scalar[dtype], MutAnyOrigin], x: Scalar[dtype]):
    y[0] = readfirstlane(x)


def test_readfirstlane_scalar_types() raises:
    """Verifies readfirstlane emits the correct LLVM intrinsic for each
    scalar type."""

    var ir_f32 = _compile_code[
        readfirstlane_kernel[.float32],
        target=MI355X_TARGET,
        emission_kind="llvm",
    ]().asm
    assert_true(
        "llvm.amdgcn.readfirstlane.f32" in ir_f32,
        "expected readfirstlane.f32 intrinsic",
    )

    var ir_u32 = _compile_code[
        readfirstlane_kernel[.uint32],
        target=MI355X_TARGET,
        emission_kind="llvm",
    ]().asm
    assert_true(
        "llvm.amdgcn.readfirstlane.i32" in ir_u32,
        "expected readfirstlane.i32 intrinsic",
    )

    var ir_f16 = _compile_code[
        readfirstlane_kernel[.float16],
        target=MI355X_TARGET,
        emission_kind="llvm",
    ]().asm
    assert_true(
        "llvm.amdgcn.readfirstlane.f16" in ir_f16,
        "expected readfirstlane.f16 intrinsic",
    )

    var ir_f64 = _compile_code[
        readfirstlane_kernel[.float64],
        target=MI355X_TARGET,
        emission_kind="llvm",
    ]().asm
    assert_true(
        "llvm.amdgcn.readfirstlane.f64" in ir_f64,
        "expected readfirstlane.f64 intrinsic",
    )


def test_readfirstlane_u64() raises:
    """Verifies that readfirstlane on a 64-bit type emits the correct
    LLVM intrinsic (llvm.amdgcn.readfirstlane.i64)."""

    def readfirstlane_u64_kernel(
        y: MutPointer[UInt64, MutAnyOrigin], x: UInt64
    ):
        y[0] = readfirstlane(x)

    var ir = _compile_code[
        readfirstlane_u64_kernel,
        target=MI355X_TARGET,
        emission_kind="llvm",
    ]().asm
    assert_true(
        "llvm.amdgcn.readfirstlane.i64" in ir,
        "expected readfirstlane.i64 intrinsic",
    )


def test_warp_id_broadcast_single_readfirstlane() raises:
    """Verifies that readfirstlane on warp_id (derived from thread_idx.x)
    does not produce two v_readfirstlane_b32 ops in the final assembly.
    The AMDGPU backend marks thread_idx.x with a known range of 0..1023,
    so thread_idx.x / WARP_SIZE fits in 32 bits and the second
    readfirstlane for the high bits should be elided."""

    def warp_id_broadcast_kernel(
        y: MutPointer[Int, MutAnyOrigin],
    ):
        var res = ufloordiv(thread_idx.x, WARP_SIZE)
        y[0] = readfirstlane(res)

    var asm = _compile_code[
        warp_id_broadcast_kernel,
        target=MI355X_TARGET,
        emission_kind="asm",
    ]().asm
    var count = asm.count("v_readfirstlane_b32")
    assert_true(
        count == 1,
        String(
            "expected exactly 1 v_readfirstlane_b32 but found ",
            count,
            "; 64-bit readfirstlane on warp_id may not be getting optimized",
        ),
    )


def main() raises:
    test_readfirstlane_scalar_types()
    test_readfirstlane_u64()
    test_warp_id_broadcast_single_readfirstlane()
