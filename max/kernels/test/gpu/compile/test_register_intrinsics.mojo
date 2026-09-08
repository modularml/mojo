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
from max.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.testing import *


def register_intrinsics():
    warpgroup_reg_alloc[64]()
    warpgroup_reg_dealloc[64]()


def test_register_intrinsics_sm80() raises:
    var asm = _compile_code[
        register_intrinsics, target=get_gpu_target["sm_80"]()
    ]().asm
    assert_false("setmaxnreg.inc.sync.aligned.u32" in asm)
    assert_false("setmaxnreg.dec.sync.aligned.u32" in asm)


def test_register_intrinsics_sm90() raises:
    var asm = _compile_code[
        register_intrinsics, target=get_gpu_target["sm_90a"]()
    ]().asm
    assert_true("setmaxnreg.inc.sync.aligned.u32" in asm)
    assert_true("setmaxnreg.dec.sync.aligned.u32" in asm)


def main() raises:
    test_register_intrinsics_sm80()
    test_register_intrinsics_sm90()
