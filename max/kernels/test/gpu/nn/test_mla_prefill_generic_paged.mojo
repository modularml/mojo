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
"""Test MLA prefill with a paged KV cache for K_rope (generic kernel).

This test exercises the production code path for DeepSeek-V2/V3
prefill: ``flare_mla_prefill`` with a ``KVCacheT`` for K_rope, dispatched
to ``mla_sm100_prefill_generic`` on B200. The existing
``test_mla.mojo``/``test_mla_prefill_qkv_fp8.mojo`` tests cover only the
contiguous K_rope path (``LayoutTensorMHAOperand``) and so do not
exercise the sub-tile TMA loops in the kernel.

The test is parameterised by ``-D page_size``; ``num_keys`` is iterated
at runtime (see ``num_keys_to_test`` in
``_paged_prefill_test_utils.mojo``). With ``BN=128`` for MLA prefill,
any config where ``page_size < 128`` makes the kernel issue
``num_rope_pages = 128/page_size`` sub-tile TMAs per tile. When
``num_keys`` doesn't cover the whole tile's row range, the trailing
sub-pages have first row ``>= num_keys`` — the kernel reads them
anyway, pulling stale data from LUT padding entries. This is the
partial-page bug described in
``docs/plans/sorted-sauteeing-snowglobe.md``.

The shared driver (``run_test_paged_prefill`` in
``_paged_prefill_test_utils.mojo``) drives the plain bf16
``flare_mla_prefill`` path used by this test and the CENG-282 vhead test;
here it runs with the DeepSeek shape (``v_head_dim == qk_nope_head_dim``,
so ``v_depth`` defaults to ``nope_depth``).

Why output-comparison alone can't detect the bug
=================================================

Stale K reads from LUT padding don't propagate to the output: the
SM100 attention softmax substitutes (does not add) ``MASK_VALUE`` for
out-of-bound score columns (``apply_mask`` / ``apply_oob_mask`` in
``attention_utils.mojo``). The OOB sub-pages always correspond to
score columns ``j >= num_keys``, which the mask unconditionally
replaces — whatever stale value was in the K register is wiped out
before softmax sees it. NVIDIA TMA's bounds-check independently
shields against any unmapped-memory fault, so even pointing the LUT
at huge page indices just returns zero-filled tiles.

Red trigger: a kernel-side `debug_assert`
==========================================

The actual signal lives inside ``mla_prefill_generic.mojo`` (and the
sibling per_token_scale / blockscale kernels): each manual K_rope
sub-tile TMA is preceded by

    debug_assert(
        kv_row + UInt32(_p * rope_sub_BN) < num_keys,
        "MLA K_rope sub-tile TMA OOB: ...",
    )

That assertion fires on configurations where the buggy comptime sub-
tile loop emits a TMA past the sequence end. With assertions enabled
(``--config=remote-b200`` enables them via ``enable_assertions=True``
in BUILD.bazel), the kernel aborts with ``CUDA_ERROR_LAUNCH_FAILED``
and the test exits non-zero. After the partial-page-aware fix, the
loop is bounded and the assert is never reached.

Configs whose loop hits a fully-OOB sub-page (and therefore fail
pre-fix): ``ps64_nk64``, ``ps32_nk96``, ``ps16_nk17``, ``ps16_nk100``.
Configs where every issued sub-page has a valid first row (and
therefore pass both pre- and post-fix): ``ps256_nk256``,
``ps128_nk128``, ``ps128_nk256``, ``ps128_nk100``, ``ps64_nk256``,
``ps64_nk100``, ``ps32_nk100``.
"""

from std.random import seed
from std.sys import get_defined_int

from max.gpu.host import DeviceContext
from max.gpu.host.info import _is_sm10x_gpu

from _paged_prefill_test_utils import (
    num_keys_to_test,
    run_test_paged_prefill,
)


# ===-----------------------------------------------------------------------===#
# Compile-time parameterisation. ``num_keys`` is iterated at runtime; only
# ``page_size`` requires a separate compilation since
# ``PagedKVCacheCollection[..., page_size]`` is parameterised on it.
# ===-----------------------------------------------------------------------===#

comptime PAGE_SIZE = get_defined_int["page_size", 256]()


def main() raises:
    with DeviceContext() as ctx:
        comptime if _is_sm10x_gpu(ctx.default_device_info):
            # Iterate over every ``num_keys`` in the shared list at
            # runtime. Re-seed per ``num_keys`` so each (page_size,
            # num_keys) configuration is independently reproducible
            # regardless of iteration order. Within a num_keys, the
            # batch_size=1 and batch_size=2 calls share the post-seed
            # RNG state (matching the pre-refactor behavior of one
            # ``seed(0)`` per binary).
            for num_keys in num_keys_to_test():
                seed(0)
                # Single-batch baseline.
                run_test_paged_prefill[
                    qkv_type=DType.bfloat16,
                    k_rope_type=DType.bfloat16,
                    output_type=DType.bfloat16,
                    depth=192,
                    num_heads=16,
                    nope_depth=128,
                    page_size=PAGE_SIZE,
                    batch_size=1,
                ](num_keys, num_keys, ctx)
                # Multi-batch — exercises cross-batch LUT layout: each
                # batch's pages occupy a distinct slice of the global
                # block array, so any incorrect LUT lookup past a
                # batch's valid page count would point to another
                # batch's data.
                run_test_paged_prefill[
                    qkv_type=DType.bfloat16,
                    k_rope_type=DType.bfloat16,
                    output_type=DType.bfloat16,
                    depth=192,
                    num_heads=16,
                    nope_depth=128,
                    page_size=PAGE_SIZE,
                    batch_size=2,
                ](num_keys, num_keys, ctx)

            # 1Q multi-tile coverage: 657 keys / 16 heads / batch 1 keeps
            # the unclamped 2Q grid at 3 * 16 = 48 blocks <= SMs/2, so
            # the dispatch heuristic picks the 1Q (num_qo=1) kernel.
            # 657 = 5 * 128 + 17 exercises the 1Q producer main loop,
            # tail, and partial last page in one shot. (The short shapes
            # in `num_keys_to_test()` also route to 1Q on B200, but only
            # cover T <= 2.)
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=192,
                num_heads=16,
                nope_depth=128,
                page_size=PAGE_SIZE,
                batch_size=1,
            ](657, 657, ctx)
            # 2Q coverage retention: 1088 keys -> 5 BM=256 tiles, so the
            # unclamped 2Q grid is 5 * 16 = 80 blocks > SMs/2 and the
            # dispatch keeps the 2Q kernel (with a partial 2Q last tile).
            seed(0)
            run_test_paged_prefill[
                qkv_type=DType.bfloat16,
                k_rope_type=DType.bfloat16,
                output_type=DType.bfloat16,
                depth=192,
                num_heads=16,
                nope_depth=128,
                page_size=PAGE_SIZE,
                batch_size=1,
            ](1088, 1088, ctx)
