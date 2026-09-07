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
"""Codegen budgets for the `range()` loops kernels are written with.

Every kernel loop goes through a `range()` iterator, so a single extra
instruction in one of them is paid per element, per kernel, everywhere. That
cost does not show up in a correctness test and only reaches a benchmark once
it has already shipped: a `select` on the loop cursor cost 6% of decode
throughput before anything caught it, because it kept LLVM from strength-
reducing the address into an induction variable.

So the shape of the loop is what is asserted here, rather than a time. A
`range()` loop that LLVM has fully understood holds one counter and one pointer,
which means:

- no `selp`, the mark of a cursor the compiler could not prove monotonic,
- no `shl` or `mul`, the mark of an address recomputed from the index instead
  of carried in a register,
- no `div.` or `rem.` anywhere in the signed forms, which terminate on a
  comparison rather than on a count they would have to derive.

Under the `select`-based cursor the strided loops ran 11 instructions per
element with the address recomputed each time. Moving the wrap flag off the
cursor rather than removing the test, which is what this file measures, carries
the address again and lands at 10. Removing the test reaches 7, which
MSTDL-3128 does; these budgets tighten to that when it lands.

The budgets are exact rather than generous, so nothing can be added on top of
the wrap test without saying so. A change that trips one should read the loop
body it prints before touching the number: an instruction traded for a `selp`
or a `shl` is the regression this test exists to catch, and a check moved off
the cursor rather than removed is what it is calibrated against.
"""

from std.compile import compile_info
from std._gpu import block_dim, thread_idx
from std._gpu.host import get_gpu_target
from std.testing import assert_equal, assert_false, assert_true

comptime _TargetType = __mlir_type.`!kgen.target`

# What a strided forward loop costs while it still tests for the wrap each trip:
# the 7 an affine cursor compiles to, plus the test and the flag it sets. The
# unstrided and descending forms come in under this on their own.
comptime FORWARD_BUDGET = 10

# The reverse iterator has no wrap to test — it stops on an inclusive bound
# instead — so it stays below the forward budget rather than above it.
comptime REVERSED_BUDGET = 8


def zero_based(data: Pointer[Float32, MutAnyOrigin], count: Int):
    for i in range(count):
        data[unsafe_offset=i] = data[unsafe_offset=i] + 1.0


def two_arg(data: Pointer[Float32, MutAnyOrigin], count: Int):
    for i in range(Int(thread_idx.x), count):
        data[unsafe_offset=i] = data[unsafe_offset=i] + 1.0


def grid_stride(data: Pointer[Float32, MutAnyOrigin], count: Int):
    """The spelling almost every kernel's outer loop has."""
    for i in range(Int(thread_idx.x), count, Int(block_dim.x)):
        data[unsafe_offset=i] = data[unsafe_offset=i] + 1.0


def grid_stride_unsigned(data: Pointer[Float32, MutAnyOrigin], count: UInt):
    for i in range(UInt(thread_idx.x), count, UInt(block_dim.x)):
        data[unsafe_offset=Int(i)] = data[unsafe_offset=Int(i)] + 1.0


def grid_stride_comptime_step(data: Pointer[Float32, MutAnyOrigin], count: Int):
    for i in range(Int(thread_idx.x), count, 128):
        data[unsafe_offset=i] = data[unsafe_offset=i] + 1.0


def grid_stride_descending(data: Pointer[Float32, MutAnyOrigin], count: Int):
    for i in range(count - 1 - Int(thread_idx.x), -1, -Int(block_dim.x)):
        data[unsafe_offset=i] = data[unsafe_offset=i] + 1.0


def grid_stride_reversed(data: Pointer[Float32, MutAnyOrigin], count: Int):
    for i in reversed(range(Int(thread_idx.x), count, Int(block_dim.x))):
        data[unsafe_offset=i] = data[unsafe_offset=i] + 1.0


def _innermost_loop_body(asm: String) raises -> String:
    """Returns the instructions between a backward branch and its label."""
    var lines = asm.split("\n")
    for i in range(len(lines)):
        var line = lines[i].strip()
        var label_start = line.find("bra")
        if label_start < 0:
            continue
        label_start = line.find("$", label_start)
        var label_end = line.find(";", label_start)
        if label_start < 0 or label_end < 0:
            continue
        var label = String(line[byte=label_start:label_end].strip(), ":")
        for j in range(i):
            if String(lines[j].strip()) != label:
                continue
            var body = String()
            for k in range(j + 1, i + 1):
                body.write(lines[k].strip(), "\n")
            return body^
    raise Error("compiled kernel holds no loop")


def _instruction_count(body: String) -> Int:
    var count = 0
    for line in body.split("\n"):
        # Labels, directives and comments are not instructions. Debug builds
        # interleave all three, so the count has to hold across them.
        if (
            line.byte_length() == 0
            or line.startswith("//")
            or line.startswith(".")
            or line.endswith(":")
        ):
            continue
        count += 1
    return count


def _assert_absent(body: String, opcode: String, name: String) raises:
    assert_false(
        opcode in body,
        String(name, ": loop body holds a `", opcode, "`\n", body),
    )


def _assert_tight_loop[
    func_type: TrivialRegisterPassable,
    //,
    func: func_type,
    *,
    target: _TargetType,
](name: String, budget: Int = FORWARD_BUDGET) raises:
    var body = _innermost_loop_body(compile_info[func, target=target]().asm)
    var count = _instruction_count(body)
    assert_true(
        count <= budget,
        String(
            name,
            ": loop body runs ",
            count,
            " instructions per element, budget ",
            budget,
            "\n",
            body,
        ),
    )
    _assert_absent(body, "selp", name)
    _assert_absent(body, "shl", name)
    _assert_absent(body, "mul", name)


def _assert_no_division[
    func_type: TrivialRegisterPassable,
    //,
    func: func_type,
    *,
    target: _TargetType,
](name: String) raises:
    var asm = compile_info[func, target=target]().asm
    assert_false("div." in asm, String(name, " divides:\n", asm))
    assert_false("rem." in asm, String(name, " takes a remainder:\n", asm))


def test_loops_stay_tight[target: _TargetType]() raises:
    _assert_tight_loop[zero_based, target=target]("zero_based")
    _assert_tight_loop[two_arg, target=target]("two_arg")
    _assert_tight_loop[grid_stride, target=target]("grid_stride")
    _assert_tight_loop[grid_stride_unsigned, target=target](
        "grid_stride_unsigned"
    )
    _assert_tight_loop[grid_stride_comptime_step, target=target](
        "grid_stride_comptime_step"
    )
    _assert_tight_loop[grid_stride_descending, target=target](
        "grid_stride_descending"
    )
    _assert_tight_loop[grid_stride_reversed, target=target](
        "grid_stride_reversed", budget=REVERSED_BUDGET
    )


def test_signed_ranges_never_divide[target: _TargetType]() raises:
    """Terminating on a comparison needs no division, so no forward form has one.

    Only the signed forms are held to that, though the unsigned one also has
    none today. MSTDL-3128 terminates on a count instead, and an unsigned count
    is the one it cannot derive without dividing — a span wider than `Int.MAX`
    can still hold few enough elements to count — so holding the unsigned form
    here would only have to be undone. `reversed()` is excluded for the reason
    that will apply: it counts, and divides once in its prologue to do it.
    """
    _assert_no_division[zero_based, target=target]("zero_based")
    _assert_no_division[two_arg, target=target]("two_arg")
    _assert_no_division[grid_stride, target=target]("grid_stride")
    _assert_no_division[grid_stride_comptime_step, target=target](
        "grid_stride_comptime_step"
    )
    _assert_no_division[grid_stride_descending, target=target](
        "grid_stride_descending"
    )


def _amd_directive(asm: String, directive: String) raises -> Int:
    var start = asm.find(directive)
    assert_true(start >= 0, String("no `", directive, "` in:\n", asm))
    var line_end = asm.find("\n", start)
    var line = asm[byte=start:line_end]
    return Int(String(line[byte = line.find(":") + 1 :].strip()))


def test_amd_register_pressure() raises:
    """AMD reports its register use, which the loop's shape moves directly.

    The `select`-based cursor needed 9 VGPRs for this loop where the carried
    pointer needs 7, and register pressure is what pushes a real kernel over
    an occupancy cliff. MSTDL-3128 drops the wrap test and reaches 5.

    TODO(MSTDL-3128): hold the AMD loop body to a shape too. Its `waitcnt` and
    exec-mask bookkeeping moves under scheduling, so counting its instructions
    needs a parser that can tell that apart from the loop's own arithmetic.
    """
    var asm = compile_info[grid_stride, target=get_gpu_target["mi355x"]()]().asm
    assert_true(
        _amd_directive(asm, ".vgpr_count") <= 7,
        String("grid_stride: too many VGPRs\n", asm),
    )
    assert_equal(_amd_directive(asm, ".vgpr_spill_count"), 0)
    assert_equal(_amd_directive(asm, ".sgpr_spill_count"), 0)


def main() raises:
    test_loops_stay_tight[get_gpu_target["sm_90"]()]()
    test_loops_stay_tight[get_gpu_target["sm_100a"]()]()
    test_signed_ranges_never_divide[get_gpu_target["sm_90"]()]()
    test_signed_ranges_never_divide[get_gpu_target["sm_100a"]()]()
    test_amd_register_pressure()
