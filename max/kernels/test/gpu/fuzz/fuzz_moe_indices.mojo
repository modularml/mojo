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
# Fuzz target: MoE `moe_create_indices` (see gpu-kernels-fuzzing-design.md).
#
# Fuzzes `num_tokens` (runtime) against a fixed `num_experts` and drives
# moe_create_indices, then copies expert_start_indices / expert_ids back to the
# host. The kernel only writes slots for *active* experts and documents a
# precondition (moe.mojo) that expert_start_indices be zero-initialized, which
# neither the kernel nor a caller satisfies here -- so when num_tokens is small
# relative to num_experts, the tail slots are uninitialized and the copy-back
# reads them. Run under `--oracle initcheck` (or `poison`) to surface it.

from max.gpu.host import DeviceContext
from std.random import random_ui64, seed
from std.sys.defines import get_defined_int

from layout import Idx, TileTensor, row_major
from layout._fillers import random
from nn.moe import moe_create_indices

from _fuzz import boundary_int, collect_args, flag, flag_int

comptime num_experts = get_defined_int["num_experts", 256]()
comptime max_tokens = get_defined_int["max_tokens", 8192]()
comptime fuzz_seed = get_defined_int["fuzz_seed", 12345]()
comptime budget = get_defined_int["budget", 16]()


@fieldwise_init
struct CaseSpec(Copyable, Movable, Writable):
    var num_tokens: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write("num_tokens=", self.num_tokens)


def gen_specs(n: Int) -> List[CaseSpec]:
    """Generates num_tokens biased toward the num_tokens << num_experts regime
    where most experts are inactive (the uninitialized-tail trigger), plus a
    general range."""
    var specs = List[CaseSpec]()
    for _ in range(n):
        var num_tokens: Int
        if Int(random_ui64(0, 1)) == 0:
            # Sparse-activation regime: many inactive experts.
            num_tokens = boundary_int(1, num_experts, 1)
        else:
            num_tokens = boundary_int(1, 4096, max_tokens)
        specs.append(CaseSpec(num_tokens))
    return specs^


def run_one_case(ctx: DeviceContext, spec: CaseSpec) raises:
    var n = spec.num_tokens

    # Host copy-back targets for the two precondition-sensitive buffers.
    var esi_host = ctx.enqueue_create_host_buffer[.uint32](num_experts + 1)
    var eids_host = ctx.enqueue_create_host_buffer[.int32](num_experts)
    var topk_host = ctx.enqueue_create_host_buffer[.uint32](n)

    var token_expert_order_dev = ctx.enqueue_create_buffer[.uint32](n)
    var expert_start_indices_dev = ctx.enqueue_create_buffer[.uint32](
        num_experts + 1
    )
    var restore_token_order_dev = ctx.enqueue_create_buffer[.uint32](n)
    var expert_ids_dev = ctx.enqueue_create_buffer[.int32](num_experts)
    var expert_usage_stats_dev = ctx.enqueue_create_buffer[.uint32](2)
    var topk_dev = ctx.enqueue_create_buffer[.uint32](n)

    var token_expert_order = TileTensor(token_expert_order_dev, row_major(n))
    var expert_start_indices = TileTensor(
        expert_start_indices_dev, row_major(num_experts + 1)
    )
    var restore_token_order = TileTensor(restore_token_order_dev, row_major(n))
    var expert_ids = TileTensor(expert_ids_dev, row_major(num_experts))
    var expert_usage_stats = TileTensor(
        expert_usage_stats_dev, row_major(Idx[2])
    )
    var topk = TileTensor(topk_dev, row_major(n))

    # Only real input: random expert ids in [0, num_experts). Fill on host.
    var topk_host_t = TileTensor(topk_host, row_major(n))
    random(topk_host_t, min=0, max=UInt32(num_experts))
    ctx.enqueue_copy(topk_dev, topk_host)

    moe_create_indices["gpu"](
        token_expert_order,
        expert_start_indices,
        restore_token_order,
        expert_ids,
        expert_usage_stats,
        topk,
        ctx,
    )

    # Uninitialized reads surface here: copy-back of partially-written buffers
    # reads the uninitialized tail slots (inactive experts) under initcheck.
    ctx.enqueue_copy(esi_host, expert_start_indices_dev)
    ctx.enqueue_copy(eids_host, expert_ids_dev)
    ctx.synchronize()

    _ = token_expert_order_dev
    _ = expert_start_indices_dev
    _ = restore_token_order_dev
    _ = expert_ids_dev
    _ = expert_usage_stats_dev
    _ = topk_dev


def main() raises:
    var args = collect_args()
    var mode = flag(args, "--mode", "fuzz")
    var the_seed = flag_int(args, "--seed", fuzz_seed)
    var the_budget = flag_int(args, "--budget", budget)
    seed(the_seed)

    if mode == "list-specs":
        var specs = gen_specs(the_budget)
        for i in range(len(specs)):
            print("FUZZ_SPEC idx=", i, "num_tokens=", specs[i].num_tokens)
        return

    if mode == "single":
        var nt = flag_int(args, "--num_tokens", 16)
        print("FUZZ_SINGLE num_tokens=", nt)
        with DeviceContext() as ctx:
            run_one_case(ctx, CaseSpec(nt))
        print("FUZZ_RESULT verdict=PASS")
        return

    print(
        "=== fuzz_moe_indices seed=",
        the_seed,
        "budget=",
        the_budget,
        "num_experts=",
        num_experts,
        "===",
    )
    var specs = gen_specs(the_budget)
    with DeviceContext() as ctx:
        for i in range(len(specs)):
            print("case", i, ":", specs[i])
            run_one_case(ctx, specs[i])
    print("=== done:", len(specs), "cases ===")
