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


from max.gpu import thread_idx
from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from max.gpu.intrinsics import ldg
from std.testing import *


def register_intrinsics(
    i8: MutPointer[Int8, MutAnyOrigin],
    ui8: MutPointer[UInt8, MutAnyOrigin],
    i16: MutPointer[Int16, MutAnyOrigin],
    ui16: MutPointer[UInt16, MutAnyOrigin],
    i32: MutPointer[Int32, MutAnyOrigin],
    ui32: MutPointer[UInt32, MutAnyOrigin],
    i64: MutPointer[Int64, MutAnyOrigin],
    ui64: MutPointer[UInt64, MutAnyOrigin],
    f32: MutPointer[Float32, MutAnyOrigin],
    f64: MutPointer[Float64, MutAnyOrigin],
):
    # Note we perform the store purely to avoid the compiler from optimizing
    # away the statements.
    var tid = thread_idx.x
    i8.store(tid, ldg(i8))
    ui8.store(tid, ldg(ui8))
    i16.store(tid, ldg(i16))
    ui16.store(tid, ldg(ui16))
    i32.store(tid, ldg(i32))
    ui32.store(tid, ldg(ui32))
    i64.store(tid, ldg(i64))
    ui64.store(tid, ldg(ui64))
    f32.store(tid, ldg(f32))
    f64.store(tid, ldg(f64))


@always_inline
def _verify_register_intrinsics(asm: StringSlice) raises -> None:
    assert_true("ld.global.nc.b8" in asm)
    assert_true("ld.global.nc.b16" in asm)
    assert_true("ld.global.nc.b32" in asm)
    assert_true("ld.global.nc.b64" in asm)
    assert_true("ld.global.nc.b32" in asm)
    assert_true("ld.global.nc.b64" in asm)


def test_register_intrinsics_sm80() raises:
    var asm = _compile_code[
        register_intrinsics, target=get_gpu_target["sm_80"]()
    ]().asm
    _verify_register_intrinsics(asm)


def test_register_intrinsics_sm90() raises:
    var asm = _compile_code[
        register_intrinsics,
        target=get_gpu_target["sm_90"](),
    ]().asm
    _verify_register_intrinsics(asm)


def main() raises:
    test_register_intrinsics_sm80()
    test_register_intrinsics_sm90()
