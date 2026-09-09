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
# Fuzz target: MHA with NullMask -- the non-causal (full-attention) path, a
# distinct masking code path from CausalPaddingMask. Memory-safety oracle
# (memcheck/redzone) by default; with --check, numerical (flash vs the naive MHA
# reference) under the `ref` oracle.

from std.math import max, min
from std.random import rand, random_ui64, seed
from std.sys.defines import get_defined_int

from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import NullMask

from _fuzz import boundary_int, collect_args, flag, flag_int, numeric_check

comptime qkv_type = DType.bfloat16
comptime depth = get_defined_int["depth", 128]()
comptime num_heads = 16
comptime group = 1
comptime kv_num_heads = num_heads // group
comptime scale = Float32(0.125)
comptime TILE = 128
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var seq_len: Int
    var num_keys: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("seq_len=", self.seq_len, " num_keys=", self.num_keys)


def gen_specs(n: Int) -> List[CaseSpec]:
    var specs = List[CaseSpec]()
    for _ in range(n):
        var num_keys = boundary_int(1, 1024, TILE)
        var seq_len: Int
        if Int(random_ui64(0, 3)) != 0:
            seq_len = 1
        else:
            seq_len = boundary_int(1, min(num_keys, 256), TILE)
        specs.append(CaseSpec(seq_len, num_keys))
    return specs^


def run_one_case(
    ctx: DeviceContext, spec: CaseSpec, check: Bool = False
) raises:
    comptime batch_size = 1
    var seq_len = spec.seq_len
    var num_keys = spec.num_keys
    var q_size = batch_size * num_heads * seq_len * depth
    var kv_size = batch_size * kv_num_heads * num_keys * depth

    var q_host = ctx.enqueue_create_host_buffer[qkv_type](q_size)
    var k_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
    var v_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
    rand(q_host.as_span())
    rand(k_host.as_span())
    rand(v_host.as_span())

    var q_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    var k_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var v_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var o_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(v_dev, v_host)

    var q = TileTensor(
        q_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )
    var k = TileTensor(
        k_dev, row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth]))
    )
    var v = TileTensor(
        v_dev, row_major((batch_size, num_keys, Idx[kv_num_heads], Idx[depth]))
    )
    var o = TileTensor(
        o_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )

    flash_attention(o, q, k, v, NullMask(), scale, ctx)
    ctx.synchronize()

    var o_ref_dev = ctx.enqueue_create_buffer[qkv_type](q_size)
    var o_ref = TileTensor(
        o_ref_dev, row_major((batch_size, seq_len, Idx[num_heads], Idx[depth]))
    )
    mha_gpu_naive(
        q,
        k,
        v,
        NullMask(),
        o_ref,
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        group,
        ctx,
    )
    ctx.synchronize()

    if check:
        var o_h = ctx.enqueue_create_host_buffer[qkv_type](q_size)
        var o_ref_h = ctx.enqueue_create_host_buffer[qkv_type](q_size)
        ctx.enqueue_copy(o_h, o_dev)
        ctx.enqueue_copy(o_ref_h, o_ref_dev)
        ctx.synchronize()
        if not numeric_check(
            o_h.as_span(), o_ref_h.as_span(), atol=2e-2, rtol=2e-2
        ):
            raise Error("flash_attention (NullMask) vs naive mismatch")

    _ = q_dev
    _ = k_dev
    _ = v_dev
    _ = o_dev
    _ = o_ref_dev


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
            print(
                "FUZZ_SPEC idx=",
                i,
                "seq_len=",
                specs[i].seq_len,
                "num_keys=",
                specs[i].num_keys,
            )
        return

    if mode == "single":
        var sl = flag_int(args, "--seq_len", 1)
        var nk = flag_int(args, "--num_keys", 128)
        print("FUZZ_SINGLE seq_len=", sl, "num_keys=", nk)
        with DeviceContext() as ctx:
            run_one_case(ctx, CaseSpec(sl, nk), check)
        print("FUZZ_RESULT verdict=PASS")
        return

    print("=== fuzz_mha_nullmask seed=", the_seed, "budget=", the_budget, "===")
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i], check)
    print("=== done:", len(specs), "cases ===")
