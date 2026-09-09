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
#
# Numeric canary -- the positive control for the numerical (`ref`) oracle, the
# analogue of fuzz_oob_canary for the memory-safety oracles. The kernel computes
# out = in * 2, but with a *deliberate, shape-dependent wrong answer*: when n is
# even it corrupts element 0. So:
#
#   * --oracle ref   -> FAIL on even n (the reference disagrees), PASS on odd n
#   * --oracle diff  -> PASS always (no crash; the wrong value is never checked)
#
# i.e. it proves the numerical oracle catches what the memory-safety/diff oracle
# cannot, and that boundary generation finds the shape-dependent numeric bug.

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.random import rand, seed
from std.sys.defines import get_defined_int

from _fuzz import boundary_int, collect_args, flag, flag_int, numeric_check

comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()
comptime BLOCK = 256


def numeric_canary_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    inp: MutPointer[Float32, MutAnyOrigin],
    n_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var n = Int(n_dev)
    var gid = global_idx.x
    if gid < n:
        var v = inp[gid] * Float32(2.0)
        # Deliberate shape-dependent wrong answer: corrupt element 0 for even n.
        if gid == 0 and n % 2 == 0:
            v = v + Float32(5.0)
        dst[gid] = v


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var n: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("n=", self.n)


def gen_specs(count: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(count):
        specs.append(CaseSpec(boundary_int(1, 4096, 256)))
    return specs^


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    var n = spec.n
    var in_host = ctx.enqueue_create_host_buffer[.float32](n)
    rand(in_host.as_span())

    var in_dev = ctx.enqueue_create_buffer[.float32](n)
    var out_dev = ctx.enqueue_create_buffer[.float32](n)
    ctx.enqueue_copy(in_dev, in_host)

    ctx.enqueue_function[numeric_canary_kernel](
        out_dev,
        in_dev,
        Int32(n),
        grid_dim=ceildiv(n, BLOCK),
        block_dim=BLOCK,
    )
    ctx.synchronize()

    if check:
        var out_h = ctx.enqueue_create_host_buffer[.float32](n)
        var ref_h = ctx.enqueue_create_host_buffer[.float32](n)
        ctx.enqueue_copy(out_h, out_dev)
        ctx.synchronize()
        var src = in_host.as_span()
        var ref_s = ref_h.as_span()
        for i in range(n):
            ref_s[i] = src[i] * Float32(2.0)
        if not numeric_check(out_h.as_span(), ref_s):
            raise Error("numeric canary mismatch")

    _ = in_dev
    _ = out_dev


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    var check = flag_int(args, "--check", 0) == 1
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print("FUZZ_SPEC idx=", i, "n=", specs[i].n)
        return

    if mode == "single":
        var n = flag_int(args, "--n", 64)
        print("FUZZ_SINGLE n=", n)
        with DeviceContext() as ctx:
            run_one_case(ctx, CaseSpec(n), check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_numeric_canary seed=", the_seed, "budget=", the_budget, "==="
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
