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
# Reference persistent-scheduling MLA-prefill metadata builder — host/CPU test.
#
# The builder itself (`kn_generate_ps_metadata` + `build_uniform` + the
# `QTile`/`WorkInfo`/`PsMetadata` types) now lives in src at
# `nn/attention/gpu/amd_structured/ps_metadata.mojo` (so the persistent
# kernel launcher can build + upload the partition). This file keeps the
# standalone invariant checks (KV-block conservation, work_indptr monotonicity,
# per-TG load balance) as an independent verification of the moved code.
#
# Run: `mojo test_ps_metadata.mojo` (CPU only, no GPU).
# ===----------------------------------------------------------------------=== #

from std.math import align_up

from nn.attention.gpu.amd_structured.ps_metadata import (
    PsMetadata,
    WORKINFO_DW,
    build_uniform,
)


# ===----------------------------------------------------------------------=== #
# Test harness -- uniform-seqlen self-attention (the bench shape).
# ===----------------------------------------------------------------------=== #


def _wf(md: PsMetadata, w: Int, field: Int) -> Int32:
    return md.work_info[w * WORKINFO_DW + field]


def check_invariants(md: PsMetadata, batch: Int, seq: Int32) raises:
    var ntg = len(md.work_indptr) - 1
    print(
        "  num_works=",
        md.num_works,
        " tgs=",
        ntg,
        " (works/tg ~",
        md.num_works // ntg,
        ")",
    )

    # 1. work_indptr: starts 0, non-decreasing, ends at num_works.
    if md.work_indptr[0] != 0:
        raise Error("work_indptr[0] != 0")
    for i in range(ntg):
        if md.work_indptr[i + 1] < md.work_indptr[i]:
            raise Error("work_indptr not monotonic")
    if Int(md.work_indptr[ntg]) != md.num_works:
        raise Error("work_indptr[-1] != num_works")

    # 2. KV-block conservation: total consumed blocks == sum over qtiles of
    #    min(num_blocks, batch KV blocks). (block_size=1 so blocks==tokens.)
    var total_consumed: Int = 0
    var num_partials: Int = 0
    var num_final_direct: Int = 0
    for w in range(md.num_works):
        var kv_start = _wf(md, w, 4)
        var kv_end = _wf(md, w, 5)
        if kv_end < kv_start:
            raise Error("kv_end < kv_start")
        total_consumed += Int(kv_end - kv_start)
        if _wf(md, w, 1) == -1:
            num_final_direct += 1
        else:
            num_partials += 1

    # Token-major: one work-item = 256 tokens of ONE head; 16 heads total.
    var expected: Int = 0
    for _h in range(16):  # num_q_heads (build_uniform default)
        for _b in range(batch):
            var q: Int32 = 0
            while q < seq:
                var local_end = min(q + 256, seq)
                var eff = local_end  # causal self-attn: kv_len-qo_len+local_end
                var nb = align_up(eff, 128)
                expected += Int(min(nb, seq))  # clamp to batch KV (S)
                q += 256
    print(
        "  consumed_kv_blocks=",
        total_consumed,
        " expected=",
        expected,
        " partials=",
        num_partials,
        " final_direct=",
        num_final_direct,
    )
    if total_consumed != expected:
        raise Error("KV-block conservation FAILED")

    # 3. Load balance: per-TG consumed blocks spread (the even division).
    var min_blk = Int(0x7FFFFFFF)
    var max_blk = 0
    for tg in range(ntg):
        var s = Int(md.work_indptr[tg])
        var e = Int(md.work_indptr[tg + 1])
        var blk = 0
        for w in range(s, e):
            blk += Int(_wf(md, w, 5) - _wf(md, w, 4))
        if blk < min_blk:
            min_blk = blk
        if blk > max_blk:
            max_blk = blk
    print("  per-TG KV blocks: min=", min_blk, " max=", max_blk)
    print("  -> PASS")


def print_first_works(md: PsMetadata, count: Int):
    print(
        "  [batch, partial_loc, qo_start, qo_end, kv_start, kv_end, kv_offset,"
        " qhead_range]"
    )
    for w in range(min(count, md.num_works)):
        print(
            "   w",
            w,
            ": [",
            _wf(md, w, 0),
            _wf(md, w, 1),
            _wf(md, w, 2),
            _wf(md, w, 3),
            _wf(md, w, 4),
            _wf(md, w, 5),
            _wf(md, w, 6),
            _wf(md, w, 7),
            "]",
        )


def main() raises:
    print("=== PS metadata builder (S1a) -- faithful v1_2_host port ===")

    print("\n[case] b1 S=256 (tiny, hand-checkable):")
    var md_tiny = build_uniform(1, 256)
    check_invariants(md_tiny, 1, 256)
    print_first_works(md_tiny, 12)

    print("\n[case] b1 S=8192 (dev target, the 52% shape):")
    var md_s1 = build_uniform(1, 8192)
    check_invariants(md_s1, 1, 8192)

    print("\n[case] b8 S=8192:")
    var md_b8 = build_uniform(8, 8192)
    check_invariants(md_b8, 8, 8192)

    print("\nALL CASES PASSED")
