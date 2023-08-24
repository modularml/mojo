# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
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

# This sample implements various memset algorithms and optimizations

from autotune import autotune_fork
from utils.list import VariadicList
from math import min, max
from time import now
from memory import memset as stdlib_memset

alias ValueType = UInt8
alias BufferPtrType = DTypePointer[DType.uint8]

alias memset_fn_type = fn (BufferPtrType, ValueType, Int) -> None


fn measure_time(
    func: memset_fn_type, size: Int, ITERS: Int, SAMPLES: Int
) -> Int:
    alias alloc_size = 1024 * 1024
    let ptr = BufferPtrType.alloc(alloc_size)

    var best = -1
    for sample in range(SAMPLES):
        let tic = now()
        for iter in range(ITERS):
            # Offset pointer to shake up cache a bit
            let offset_ptr = ptr.offset((iter * 128) & 1024)

            # Just in case compiler will try to outsmart us and avoid repeating
            # memset, change the value we're filling with
            let v = ValueType(iter&255)

            # Actually call the memset function
            func(offset_ptr, v.value, size)

        let toc = now()
        if best < 0 or toc - tic < best:
            best = toc - tic

    ptr.free()
    return best


alias MULT = 2_000


fn visualize_result(size: Int, result: Int):
    print_no_newline("Size: ")
    if size < 10:
        print_no_newline(" ")
    print_no_newline(size, "  |")
    for _ in range(result // MULT):
        print_no_newline("*")
    print()


fn benchmark(func: memset_fn_type, title: StringRef):
    print("\n=====================")
    print(title)
    print("---------------------\n")

    alias benchmark_iterations = 30 * MULT
    alias warmup_samples = 10
    alias benchmark_samples = 1000

    # Warmup
    for size in range(35):
        _ = measure_time(func, size, benchmark_iterations, warmup_samples)

    # Actual run
    for size in range(35):
        let result = measure_time(
            func, size, benchmark_iterations, benchmark_samples
        )

        visualize_result(size, result)


@always_inline
fn overlapped_store[
    width: Int
](ptr: BufferPtrType, value: ValueType, count: Int):
    let v = SIMD.splat[DType.uint8, width](value)
    ptr.simd_store[width](v)
    ptr.simd_store[width](count - width, v)


fn memset_manual(ptr: BufferPtrType, value: ValueType, count: Int):
    if count < 32:
        if count < 5:
            if count == 0:
                return
            # 0 < count <= 4
            ptr.store(0, value)
            ptr.store(count - 1, value)
            if count <= 2:
                return
            ptr.store(1, value)
            ptr.store(count - 2, value)
            return

        if count <= 16:
            if count >= 8:
                # 8 <= count < 16
                overlapped_store[8](ptr, value, count)
                return
            # 4 < count < 8
            overlapped_store[4](ptr, value, count)
            return

        # 16 <= count < 32
        overlapped_store[16](ptr, value, count)
    else:
        # 32 < count
        memset_system(ptr, value, count)


fn memset_system(ptr: BufferPtrType, value: ValueType, count: Int):
    stdlib_memset(ptr, value.value, count)


fn memset_manual_2(ptr: BufferPtrType, value: ValueType, count: Int):
    if count < 32:
        if count >= 16:
            # 16 <= count < 32
            overlapped_store[16](ptr, value, count)
            return

        if count < 5:
            if count == 0:
                return
            # 0 < count <= 4
            ptr.store(0, value)
            ptr.store(count - 1, value)
            if count <= 2:
                return
            ptr.store(1, value)
            ptr.store(count - 2, value)
            return

        if count >= 8:
            # 8 <= count < 16
            overlapped_store[8](ptr, value, count)
            return
        # 4 < count < 8
        overlapped_store[4](ptr, value, count)

    else:
        # 32 < count
        memset_system(ptr, value, count)


@adaptive
@always_inline
fn memset_impl_layer[
    lower: Int, upper: Int
](ptr: BufferPtrType, value: ValueType, count: Int):
    @parameter
    if lower == -100 and upper == 0:
        pass
    elif lower == 0 and upper == 4:
        ptr.store(0, value)
        ptr.store(count - 1, value)
        if count <= 2:
            return
        ptr.store(1, value)
        ptr.store(count - 2, value)
    elif lower == 4 and upper == 8:
        overlapped_store[4](ptr, value, count)
    elif lower == 8 and upper == 16:
        overlapped_store[8](ptr, value, count)
    elif lower == 16 and upper == 32:
        overlapped_store[16](ptr, value, count)
    elif lower == 32 and upper == 100:
        memset_system(ptr, value, count)
    else:
        constrained[False]()


@adaptive
@always_inline
fn memset_impl_layer[
    lower: Int, upper: Int
](ptr: BufferPtrType, value: ValueType, count: Int):
    alias cur: Int
    autotune_fork[Int, 0, 4, 8, 16, 32 -> cur]()

    constrained[cur > lower]()
    constrained[cur < upper]()

    if count > cur:
        memset_impl_layer[max(cur, lower), upper](ptr, value, count)
    else:
        memset_impl_layer[lower, min(cur, upper)](ptr, value, count)


@adaptive
@always_inline
fn memset_impl_layer[
    lower: Int, upper: Int
](ptr: BufferPtrType, value: ValueType, count: Int):
    alias cur: Int
    autotune_fork[Int, 0, 4, 8, 16, 32 -> cur]()

    constrained[cur > lower]()
    constrained[cur < upper]()

    if count <= cur:
        memset_impl_layer[lower, min(cur, upper)](ptr, value, count)
    else:
        memset_impl_layer[max(cur, lower), upper](ptr, value, count)


fn memset_evaluator(funcs: Pointer[memset_fn_type], size: Int) -> Int:
    # This size is picked at random, in real code we could use a real size
    # distribution here.
    let size_to_optimize_for = 17
    print("Optimizing for size: ", size_to_optimize_for)

    var best_idx: Int = -1
    var best_time: Int = -1

    alias eval_iterations = MULT
    alias eval_samples = 500

    # Find the function that's the fastest on the size we're optimizing for
    for f_idx in range(size):
        let func = funcs.load(f_idx)
        let cur_time = measure_time(
            func, size_to_optimize_for, eval_iterations, eval_samples
        )
        if best_idx < 0:
            best_idx = f_idx
            best_time = cur_time
        if best_time > cur_time:
            best_idx = f_idx
            best_time = cur_time

    return best_idx


fn main():
    # CHECK: Manual memset
    # CHECK: System memset
    benchmark(memset_manual, "Manual memset")
    benchmark(memset_system, "System memset")
    # CHECK: Manual memset v2
    benchmark(memset_manual_2, "Manual memset v2")
    benchmark(memset_system, "Mojo system memset")
    # CHECK: Mojo autotune memset
    benchmark(memset_manual, "Mojo manual memset")
    benchmark(memset_manual_2, "Mojo manual memset v2")
    benchmark(memset_system, "Mojo system memset")
