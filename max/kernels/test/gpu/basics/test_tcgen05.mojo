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
from max.gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptor
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_cp,
    tcgen05_dealloc,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_store_wait,
)
from layout import IntTuple, Layout, LayoutTensor
from std.testing import assert_true


def alloc_test_fn[cta_group: Int32]():
    var ptr_tmem_addr = MutPointer[
        UInt32, MutAnyOrigin, address_space=.SHARED
    ].unsafe_dangling()
    var num_cols: UInt32 = 32
    tcgen05_alloc[cta_group](ptr_tmem_addr, num_cols)


def test_tcgen05_alloc() raises:
    var asm1 = _compile_code[
        alloc_test_fn[1],
        target=get_gpu_target["sm_100a"](),
    ]().asm
    var asm2 = _compile_code[
        alloc_test_fn[2],
        target=get_gpu_target["sm_100a"](),
    ]().asm
    assert_true(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32" in asm1
    )
    assert_true(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32" in asm2
    )


def alloc_dealloc_test_fn():
    var ptr_tmem_addr = MutPointer[
        UInt32, MutAnyOrigin, address_space=.SHARED
    ].unsafe_dangling()
    var tmem_addr: UInt32 = 0
    var num_cols: UInt32 = 32
    tcgen05_alloc[1](ptr_tmem_addr, num_cols)
    tcgen05_release_allocation_lock[1]()
    tcgen05_dealloc[1](tmem_addr, num_cols)


def test_tcgen05_dealloc() raises:
    var asm = _compile_code[
        alloc_dealloc_test_fn,
        target=get_gpu_target["sm_100a"](),
    ]().asm
    assert_true(
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" in asm
    )
    assert_true("tcgen05.dealloc.cta_group::1.sync.aligned.b32" in asm)


def ld_test_fn[repeat: Int]():
    var ptr_tmem_addr = MutPointer[
        UInt32, MutAnyOrigin, address_space=.SHARED
    ].unsafe_dangling()
    var num_cols: UInt32 = 32
    tcgen05_alloc[1](ptr_tmem_addr, num_cols)
    var tmem_addr = ptr_tmem_addr[0]
    _ = tcgen05_ld[
        datapaths=32,
        bits=32,
        repeat=repeat,
        dtype=DType.float32,
        pack=False,
        width=repeat,
    ](tmem_addr)
    tcgen05_load_wait()
    tcgen05_dealloc[1](tmem_addr, num_cols)


def test_tcgen05_ld() raises:
    var asm_64 = _compile_code[
        ld_test_fn[64],
        target=get_gpu_target["sm_100a"](),
    ]().asm
    assert_true("tcgen05.ld.sync.aligned.32x32b.x64.b32" in asm_64)
    assert_true("tcgen05.wait::ld.sync.aligned;" in asm_64)

    var asm_1 = _compile_code[
        ld_test_fn[1],
        target=get_gpu_target["sm_100a"](),
    ]().asm
    assert_true("tcgen05.ld.sync.aligned.32x32b.x1.b32" in asm_1)
    assert_true("tcgen05.wait::ld.sync.aligned;" in asm_1)


def st_test_fn():
    var ptr_tmem_addr = MutPointer[
        UInt32, MutAnyOrigin, address_space=.SHARED
    ].unsafe_dangling()
    var num_cols: UInt32 = 32
    tcgen05_alloc[1](ptr_tmem_addr, num_cols)
    var tmem_addr = ptr_tmem_addr[0]
    var data = Array[Float32, 64](fill={})
    tcgen05_st[
        datapaths=32,
        bits=32,
        repeat=64,
        pack=False,
    ](tmem_addr, data)
    tcgen05_store_wait()
    tcgen05_dealloc[1](tmem_addr, num_cols)


def test_tcgen05_st() raises:
    var asm = _compile_code[
        st_test_fn,
        target=get_gpu_target["sm_100a"](),
    ]().asm
    assert_true("tcgen05.st.sync.aligned.32x32b.x64.b32" in asm)
    assert_true("tcgen05.wait::st.sync.aligned;" in asm)


def cp_test_fn():
    var ptr_tmem_addr = MutPointer[
        UInt32, MutAnyOrigin, address_space=.SHARED
    ].unsafe_dangling()
    var num_cols: UInt32 = 32
    tcgen05_alloc[1](ptr_tmem_addr, num_cols)
    var tmem_addr = ptr_tmem_addr[0]

    var smem_tile = LayoutTensor[
        .float32,
        Layout(IntTuple(32, 32)),
        MutAnyOrigin,
        address_space=.SHARED,
        alignment=128,
    ].stack_allocation()

    var s_desc = MMASmemDescriptor.create[0, 0](smem_tile.ptr)

    tcgen05_cp[
        cta_group=1,
        datapaths=128,
        bits=256,
        src_fmt="b6x16_p32",
        dst_fmt="b8x16",
        multicast="warpx2::01_23",
    ](tmem_addr, s_desc)


def test_tcgen05_cp() raises:
    var asm = _compile_code[
        cp_test_fn,
        target=get_gpu_target["sm_100a"](),
    ]().asm
    assert_true(
        "tcgen05.cp.cta_group::1.128x256b.warpx2::01_23.b8x16.b6x16_p32" in asm
    )


def main() raises:
    test_tcgen05_alloc()
    test_tcgen05_dealloc()
    test_tcgen05_ld()
    test_tcgen05_st()
    test_tcgen05_cp()
