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

from std.math import exp

from max.gpu import (
    thread_idx,
    block_dim,
    grid_dim,
    lane_id,
)
from max.gpu.sync import (
    AMDScheduleBarrierMask,
    barrier,
    schedule_barrier,
    schedule_group_barrier,
    s_waitcnt,
    s_waitcnt_barrier,
)
from max.gpu.globals import WARP_SIZE
from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from max.gpu.intrinsics import (
    ds_read_tr8_b64,
    ds_read_tr16_b64,
    permlane_shuffle,
    permlane_swap,
)
from max.gpu.primitives.warp import (
    shuffle_down,
    shuffle_idx,
    shuffle_up,
    shuffle_xor,
)
from std.benchmark import keep

comptime MI300X_TARGET = get_gpu_target["mi300x"]()
comptime MI355X_TARGET = get_gpu_target["mi355x"]()

comptime FULL_MASK_AMD = 2**WARP_SIZE - 1


def kernel(x: MutPointer[Int, MutAnyOrigin]):
    x[0] = thread_idx.x


def kernel_laneid(x: MutPointer[Int, MutAnyOrigin]):
    x[0] = lane_id()


def kernel_exp[
    dtype: DType
](x: MutPointer[Scalar[dtype], MutAnyOrigin]) where dtype.is_floating_point():
    x[0] = exp(x[0])


def kernel_shuffle_down(x: MutPointer[UInt32, MutAnyOrigin]):
    var val = x[0]
    var mask = UInt(FULL_MASK_AMD)
    var offset = x[0]
    x[0] = shuffle_down(mask, val, offset)


def kernel_shuffle_up(x: MutPointer[UInt32, MutAnyOrigin]):
    var val = x[0]
    var mask = UInt(FULL_MASK_AMD)
    var offset = x[0]
    x[0] = shuffle_up(mask, val, offset)


def kernel_shuffle_xor(x: MutPointer[UInt32, MutAnyOrigin]):
    var val = x[0]
    var mask = UInt(FULL_MASK_AMD)
    var offset = x[0]
    x[0] = shuffle_xor(mask, val, offset)


def kernel_shuffle_idx(x: MutPointer[UInt32, MutAnyOrigin]):
    var val = x[0]
    var mask = UInt(FULL_MASK_AMD)
    var offset = x[0]
    x[0] = shuffle_idx(mask, val, offset)


def kernel_cast[
    dtype: DType, target: DType
](
    x: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    y: MutPointer[Scalar[target], MutAnyOrigin],
):
    y[0] = x[0].cast[target]()


def parametric[
    f: def(MutPointer[Int, MutAnyOrigin]) thin -> None
](ptr: MutPointer[Int, MutAnyOrigin]):
    f(ptr)


# from https://rocm.blogs.amd.com/software-tools-optimization/amdgcn-isa/README.html#naive-load-and-store
def load_store(
    n: Int,
    input: ImmPointer[Float32, ImmutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
):
    var tid = thread_idx.x + block_dim.x * grid_dim.x
    output[tid] = input[tid]


# CHECK-LABEL: test_shuffle_compile
def test_shuffle_compile() raises:
    print("== test_shuffle_compile")
    # CHECK: %3 = load i32, ptr addrspace(1) %2, align 4, !amdgpu.noclobber !2
    # CHECK: %4 = tail call range(i32 0, 33) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
    # CHECK: %5 = tail call range(i32 0, 65) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %4)
    # CHECK: %6 = add i32 %5, %3
    # CHECK: %7 = icmp ult i32 %6, 64
    # CHECK: %8 = select i1 %7, i32 %3, i32 0
    # CHECK: %9 = add i32 %8, %5
    # CHECK: %10 = shl i32 %9, 2
    # CHECK: %11 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %10, i32 %3)
    print(
        _compile_code[
            kernel_shuffle_down, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )

    # CHECK: %3 = load i32, ptr addrspace(1) %2, align 4, !amdgpu.noclobber !2
    # CHECK: %4 = tail call range(i32 0, 33) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
    # CHECK: %5 = tail call range(i32 0, 65) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %4)
    # CHECK: %6 = sub i32 %5, %3
    # CHECK: %7 = and i32 %5, 64
    # CHECK: %8 = icmp slt i32 %6, %7
    # CHECK: %9 = select i1 %8, i32 %5, i32 %6
    # CHECK: %10 = shl i32 %9, 2
    # CHECK: %11 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %10, i32 %3)
    print(
        _compile_code[
            kernel_shuffle_up, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )

    # CHECK: %3 = load i32, ptr addrspace(1) %2, align 4, !amdgpu.noclobber !2
    # CHECK: %4 = tail call range(i32 0, 33) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
    # CHECK: %5 = tail call range(i32 0, 65) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %4)
    # CHECK: %6 = xor i32 %5, %3
    # CHECK: %7 = and i32 %5, 64
    # CHECK: %8 = add nuw nsw i32 %7, 64
    # CHECK: %9 = icmp ult i32 %6, %8
    # CHECK: %10 = select i1 %9, i32 %6, i32 %5
    # CHECK: %11 = shl i32 %10, 2
    # CHECK: %12 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %11, i32 %3)
    print(
        _compile_code[
            kernel_shuffle_xor, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )

    # CHECK: %3 = load i32, ptr addrspace(1) %2, align 4, !amdgpu.noclobber !2
    # CHECK: %4 = tail call range(i32 0, 33) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
    # CHECK: %5 = tail call range(i32 0, 65) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %4)
    # CHECK: %6 = and i32 %5, 64
    # CHECK: %7 = or i32 %6, %3
    # CHECK: %8 = shl i32 %7, 2
    # CHECK: %9 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %8, i32 %3)
    print(
        _compile_code[
            kernel_shuffle_idx, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )


# CHECK-LABEL: test_cast_fp32_bf16_compile
def test_cast_fp32_bf16_compile() raises:
    print("== test_cast_fp32_bf16_compile")

    # CHECK: tail call i64 asm "v_cmp_u_f32 $0, $1, $1"
    # CHECK: tail call i32 asm "v_bfe_u32 $0, $1, 16, 1"
    # CHECK: tail call i32 asm "v_add3_u32 $0, $1, $2, $3"
    # CHECK: tail call i32 asm "v_cndmask_b32 $0, $1, $2, $3"
    print(
        _compile_code[
            kernel_cast[.float32, DType.bfloat16],
            target=MI300X_TARGET,
            emission_kind="llvm-opt",
        ]()
    )


# CHECK-LABEL: test_exp_f32_compile
def test_exp_f32_compile() raises:
    print("== test_exp_f32_compile")

    # CHECK: tail call float @llvm.amdgcn.exp2.f32(float %4)
    print(
        _compile_code[
            kernel_exp[.float32],
            target=MI300X_TARGET,
            emission_kind="llvm-opt",
        ]()
    )


# CHECK-LABEL: test_exp_f16_compile
def test_exp_f16_compile() raises:
    print("== test_exp_f16_compile")

    # CHECK: tail call half @llvm.amdgcn.exp2.f16(half %4)
    print(
        _compile_code[
            kernel_exp[.float16],
            target=MI300X_TARGET,
            emission_kind="llvm-opt",
        ]()
    )


# CHECK-LABEL: test_laneid_compile
def test_laneid_compile() raises:
    print("== test_laneid_compile")

    # CHECK: %3 = tail call range(i32 0, 33) i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
    # CHECK: %4 = tail call range(i32 0, 65) i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %3)
    print(
        _compile_code[
            kernel_laneid, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )


# CHECK-LABEL: test_barrier_compile
def test_barrier_compile() raises:
    print("== test_barrier_compile")

    # CHECK: fence syncscope("workgroup") release
    # CHECK: tail call void @llvm.amdgcn.s.barrier()
    # CHECK: syncscope("workgroup") acquire
    print(
        _compile_code[barrier, target=MI300X_TARGET, emission_kind="llvm-opt"]()
    )


# CHECK-LABEL: test_threadid_compile
def test_threadid_compile() raises:
    print("== test_threadid_compile")

    # CHECK: .amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"
    # CHECK: s_waitcnt lgkmcnt
    print(_compile_code[kernel, target=MI300X_TARGET]())
    # CHECK: .amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx942"
    # CHECK: s_waitcnt lgkmcnt
    print(_compile_code[parametric[kernel], target=MI300X_TARGET]())
    # CHECK: ; ModuleID =
    # CHECK: llvm.amdgcn.workitem.id.x
    print(
        _compile_code[
            parametric[kernel],
            target=MI300X_TARGET,
            emission_kind="llvm",
        ]()
    )

    # CHECK-LABEL: @test_compile_gcn_load_store_
    # CHECK: llvm.amdgcn.workitem.id.x
    # CHECK: %[[VAR:.*]] = tail call {{.*}}ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
    # CHECK: getelementptr inbounds nuw i8, ptr addrspace(4) %[[VAR]], i64 12
    print(
        _compile_code[
            load_store, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )

    _ = _compile_code[load_store, target=MI300X_TARGET]()


# CHECK-LABEL: test_schedule_barrier_compile
def test_schedule_barrier_compile() raises:
    print("== test_schedule_barrier_compile")

    def schedule_kernel():
        schedule_barrier(AMDScheduleBarrierMask.NONE)
        schedule_barrier(AMDScheduleBarrierMask.ALL_ALU)
        schedule_barrier(AMDScheduleBarrierMask.VALU)
        schedule_barrier(AMDScheduleBarrierMask.SALU)
        schedule_barrier(AMDScheduleBarrierMask.MFMA)
        schedule_barrier(AMDScheduleBarrierMask.ALL_VMEM)
        schedule_barrier(AMDScheduleBarrierMask.VMEM_READ)
        schedule_barrier(AMDScheduleBarrierMask.VMEM_WRITE)
        schedule_barrier(AMDScheduleBarrierMask.ALL_DS)
        schedule_barrier(AMDScheduleBarrierMask.DS_READ)
        schedule_barrier(AMDScheduleBarrierMask.DS_WRITE)
        schedule_barrier(AMDScheduleBarrierMask.TRANS)

    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 0)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 1)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 2)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 4)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 8)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 16)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 32)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 64)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 128)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 256)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 512)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 1024)
    print(
        _compile_code[
            schedule_kernel, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )


# CHECK-LABEL: test_schedule_group_barrier_compile
def test_schedule_group_barrier_compile() raises:
    print("== test_schedule_group_barrier_compile")

    def schedule_kernel():
        schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 10, 0)
        schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 10, 1)
        schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 11, 10)
        schedule_group_barrier(AMDScheduleBarrierMask.TRANS, 11, 10)

    # CHECK: tail call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 10, i32 0)
    # CHECK: tail call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 10, i32 1)
    # CHECK: tail call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 11, i32 10)
    # CHECK: tail call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 11, i32 10)
    print(
        _compile_code[
            schedule_kernel, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )


# CHECK-LABEL: test_ds_read_tr16_b64_compile
def test_ds_read_tr16_b64_compile() raises:
    print("== test_ds_read_tr16_b64_compile")

    def test_kernel[dtype: DType]():
        var x = MutPointer[
            Scalar[dtype], MutAnyOrigin, address_space=.SHARED
        ].unsafe_dangling()
        var y = ds_read_tr16_b64(x)
        y[0] = y[0] + 1
        x[0] = y[0]

    # CHECK: ds_read_b64_tr_b16 v[0:1], v2
    print(
        _compile_code[
            test_kernel[.float16],
            target=MI355X_TARGET,
        ]()
    )
    # CHECK: ds_read_b64_tr_b16 v[0:1], v2
    print(
        _compile_code[
            test_kernel[.bfloat16],
            target=MI355X_TARGET,
        ]()
    )
    # CHECK: ds_read_b64_tr_b16 v[0:1], v2
    print(
        _compile_code[
            test_kernel[.int16],
            target=MI355X_TARGET,
        ]()
    )
    # CHECK: ds_read_b64_tr_b16 v[0:1], v2
    print(
        _compile_code[
            test_kernel[.uint16],
            target=MI355X_TARGET,
        ]()
    )


# CHECK-LABEL: test_ds_read_tr8_b64_compile
def test_ds_read_tr8_b64_compile() raises:
    print("== test_ds_read_tr8_b64_compile")

    def test_kernel[dtype: DType]():
        var x = MutPointer[
            Scalar[dtype], MutAnyOrigin, address_space=.SHARED
        ].unsafe_dangling()
        var y = ds_read_tr8_b64(x)
        y[0] = y[0] + 1
        x[0] = y[0]

    # CHECK: ds_read_b64_tr_b8 v[0:1], v2
    print(
        _compile_code[
            test_kernel[.uint8],
            target=MI355X_TARGET,
        ]()
    )
    # CHECK: ds_read_b64_tr_b8 v[0:1], v2
    print(
        _compile_code[
            test_kernel[.int8],
            target=MI355X_TARGET,
        ]()
    )


# CHECK-LABEL: test_permlane_compile
def test_permlane_compile() raises:
    print("== test_permlane_compile")

    def test_kernel[dtype: DType]():
        var val = Scalar[dtype](lane_id())
        var val_perm_16 = permlane_shuffle[16](val)
        var val_perm_32 = permlane_shuffle[32](val)
        var val_swap_16 = permlane_swap[16](val, val * val)
        var val_swap_32 = permlane_swap[32](val, val * val)
        keep(val_perm_16)
        keep(val_perm_32)
        keep(val_swap_16)
        keep(val_swap_32)

    # CHECK: v_permlane16_swap_b32_e32 v{{.*}} v{{.*}}
    # CHECK: v_permlane32_swap_b32_e32 v{{.*}} v{{.*}}
    # CHECK: v_permlane16_swap_b32_e32 v{{.*}} v{{.*}}
    # CHECK: v_permlane32_swap_b32_e32 v{{.*}} v{{.*}}
    print(
        _compile_code[
            test_kernel[.float32],
            target=MI355X_TARGET,
        ]()
    )


# CHECK-LABEL: test_waitcnt_compile
def test_waitcnt_compile() raises:
    print("== test_waitcnt_compile")

    def test_kernel_1():
        s_waitcnt[vmcnt=11]()

    def test_kernel_2():
        s_waitcnt[vmcnt=2, lgkmcnt=2]()

    def test_kernel_3():
        s_waitcnt[lgkmcnt=2]()

    def test_kernel_4():
        s_waitcnt_barrier[vmcnt=12, lgkmcnt=9]()

    print("== test_kernel_1")
    # CHECK-LABEL: test_kernel_1
    # CHECK: s_waitcnt vmcnt(11)
    print(
        _compile_code[
            test_kernel_1,
            target=MI355X_TARGET,
        ]()
    )

    print("== test_kernel_2")
    # CHECK-LABEL: test_kernel_2
    # CHECK: s_waitcnt vmcnt(2) lgkmcnt(2)
    print(
        _compile_code[
            test_kernel_2,
            target=MI355X_TARGET,
        ]()
    )

    print("== test_kernel_3")
    # CHECK-LABEL: test_kernel_3
    # CHECK: s_waitcnt lgkmcnt(2)
    print(
        _compile_code[
            test_kernel_3,
            target=MI355X_TARGET,
        ]()
    )

    print("== test_kernel_4")
    # CHECK-LABEL: test_kernel_4
    # CHECK: s_waitcnt vmcnt(12) lgkmcnt(9)
    # CHECK: s_barrier
    print(
        _compile_code[
            test_kernel_4,
            target=MI355X_TARGET,
        ]()
    )


# CHECK-LABEL: test_nt_load_compile
def test_nt_load_compile() raises:
    print("== test_nt_load_compile")

    def kernel(x: ImmPointer[Float32, ImmutAnyOrigin]):
        var ptr = x + thread_idx.x * 4
        var val = ptr.load[width=4, non_temporal=True]()
        keep(val)

    # CHECK: global_load_dwordx4 {{.*}} nt
    print(_compile_code[kernel, target=MI355X_TARGET]())


# CHECK-LABEL: test_nt_store_compile
def test_nt_store_compile() raises:
    print("== test_nt_store_compile")

    def kernel(
        x: ImmPointer[Float32, ImmutAnyOrigin],
        y: MutPointer[Float32, MutAnyOrigin],
    ):
        var ptr_in = x + thread_idx.x * 4
        var ptr_out = y + thread_idx.x * 4
        var val = ptr_in.load[width=4]()
        ptr_out.store[non_temporal=True](val)

    # CHECK: global_store_dwordx4 {{.*}} nt
    print(_compile_code[kernel, target=MI355X_TARGET]())


def main() raises:
    test_shuffle_compile()
    test_cast_fp32_bf16_compile()
    test_exp_f32_compile()
    test_exp_f16_compile()
    test_laneid_compile()
    test_barrier_compile()
    test_threadid_compile()
    test_schedule_barrier_compile()
    test_schedule_group_barrier_compile()
    test_ds_read_tr16_b64_compile()
    test_ds_read_tr8_b64_compile()
    test_permlane_compile()
    test_waitcnt_compile()
    test_nt_load_compile()
    test_nt_store_compile()
