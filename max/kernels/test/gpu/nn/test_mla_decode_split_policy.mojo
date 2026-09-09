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

"""Host-side pins for the MLA decode split-K partition choice and the runtime
head-count binding that reaches it.

The split count is numerically inert: every bucket reduces to the same answer,
so choosing a worse one costs wall time and nothing else. No GPU test can see
that, which leaves the policy free to drift silently. The binding is invisible
for the opposite reason -- the GPU tests instantiate the comptime dispatch
directly and never cross the runtime enumeration at all. These pins assert both
decisions on the host, where they are just arithmetic.
"""

from std.math import ceildiv

from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    _NUM_PARTITION_BUCKETS,
    _bucket_num_partitions,
    _compute_num_partitions,
    _get_partition_bucket,
    compute_mla_dispatch_scalars,
    compute_mla_dispatch_scalars_runtime,
)

comptime SM_COUNT = 148
comptime HALF_SMS = SM_COUNT // 2


def _expect(what: StringLiteral, got: Int, want: Int) raises:
    if got != want:
        raise Error(what, ": got ", got, ", want ", want)


def _is_bucket(num_partitions: Int) -> Bool:
    comptime for i in range(_NUM_PARTITION_BUCKETS):
        if _get_partition_bucket[HALF_SMS, i]() == num_partitions:
            return True
    return False


def _expect_bucket(
    what: StringLiteral, num_partitions: Int, batch: Int, cache: Int, q: Int
) raises:
    if not _is_bucket(num_partitions):
        raise Error(
            what,
            " chose num_partitions=",
            num_partitions,
            " which has no compiled combine kernel (batch=",
            batch,
            " cache=",
            cache,
            " q=",
            q,
            ")",
        )


def test_bucket_tables_agree() raises:
    """The bucket table is written out three times and nothing keeps the copies
    in sync: as a literal in `_get_partition_bucket`, again in
    `_bucket_num_partitions`, and as the count `_NUM_PARTITION_BUCKETS`. A
    divergence reproduces the unreduced-output bug the dispatch now raises on.

    These properties pin the three to each other without restating the table,
    so the test cannot drift along with it.
    """
    comptime for i in range(_NUM_PARTITION_BUCKETS):
        var b = _get_partition_bucket[HALF_SMS, i]()
        # A bucket must map to itself. If `_bucket_num_partitions` is missing
        # this entry it rounds up to the next one instead.
        _expect(
            "bucket does not map to itself",
            _bucket_num_partitions[HALF_SMS](b),
            b,
        )

    comptime for i in range(1, _NUM_PARTITION_BUCKETS):
        var prev = _get_partition_bucket[HALF_SMS, i - 1]()
        var b = _get_partition_bucket[HALF_SMS, i]()
        if prev >= b:
            raise Error(
                "bucket table is not strictly ascending at index ",
                i,
                ": ",
                prev,
                " >= ",
                b,
            )
        # One past the previous bucket must land on this one. Catches the other
        # direction: an entry present only in `_bucket_num_partitions`.
        _expect(
            "wrong bucket for prev+1",
            _bucket_num_partitions[HALF_SMS](prev + 1),
            b,
        )

    # A stale count leaves the last checked entry short of the SM-adaptive one.
    _expect(
        "last bucket is not half_sms",
        _get_partition_bucket[HALF_SMS, _NUM_PARTITION_BUCKETS - 1](),
        HALF_SMS,
    )

    _expect("floor", _bucket_num_partitions[HALF_SMS](0), 1)
    _expect("floor", _bucket_num_partitions[HALF_SMS](1), 1)
    _expect(
        "clamp above the top bucket",
        _bucket_num_partitions[HALF_SMS](HALF_SMS + 1),
        HALF_SMS,
    )


def _check_every_path(batch: Int, cache: Int, q: Int, page: Int) raises:
    """Every head count, both split-K policies, both floor policies."""
    for r in range(2):
        var relax = r == 1
        for c in range(2):
            var cost = c == 1
            _expect_bucket(
                "h8",
                _compute_num_partitions[8, True, HALF_SMS, False](
                    batch,
                    cache,
                    q,
                    page,
                    SM_COUNT,
                    relax_split_floor=relax,
                    cost_optimal_split=cost,
                ),
                batch,
                cache,
                q,
            )
            _expect_bucket(
                "h8 folded",
                _compute_num_partitions[8, True, HALF_SMS, True](
                    batch,
                    cache,
                    q,
                    page,
                    SM_COUNT,
                    relax_split_floor=relax,
                    cost_optimal_split=cost,
                ),
                batch,
                cache,
                q,
            )
            _expect_bucket(
                "h8 bf16 kv",
                _compute_num_partitions[8, False, HALF_SMS, False](
                    batch,
                    cache,
                    q,
                    page,
                    SM_COUNT,
                    relax_split_floor=relax,
                    cost_optimal_split=cost,
                ),
                batch,
                cache,
                q,
            )
            _expect_bucket(
                "h16",
                _compute_num_partitions[16, True, HALF_SMS, False](
                    batch,
                    cache,
                    q,
                    page,
                    SM_COUNT,
                    relax_split_floor=relax,
                    cost_optimal_split=cost,
                ),
                batch,
                cache,
                q,
            )
            _expect_bucket(
                "h64",
                _compute_num_partitions[64, True, HALF_SMS, False](
                    batch,
                    cache,
                    q,
                    page,
                    SM_COUNT,
                    relax_split_floor=relax,
                    cost_optimal_split=cost,
                ),
                batch,
                cache,
                q,
            )
            _expect_bucket(
                "h128",
                _compute_num_partitions[128, True, HALF_SMS, False](
                    batch,
                    cache,
                    q,
                    page,
                    SM_COUNT,
                    relax_split_floor=relax,
                    cost_optimal_split=cost,
                ),
                batch,
                cache,
                q,
            )


def test_every_split_has_a_combine_kernel() raises:
    """Host mirror of the dispatch's runtime guard.

    A split outside the bucket set launches the decode and never reduces its
    partial outputs, which is indistinguishable from a correct result. The
    guard catches that at runtime on the shapes something happens to run; this
    catches it over a sweep, including prime lengths no production config uses.
    """
    var batches = [1, 2, 3, 5, 8, 13, 16, 31, 32, 64, 128]
    var caches = [251, 512, 1021, 1536, 2048, 3072, 4099]
    var qs = [1, 2, 3, 5, 6, 7]
    var pages = [64, 128]
    for bi in range(len(batches)):
        for ci in range(len(caches)):
            for qi in range(len(qs)):
                for pi in range(len(pages)):
                    _check_every_path(
                        batches[bi], caches[ci], qs[qi], pages[pi]
                    )


def _np_h8(
    batch: Int, cache: Int, q: Int, page: Int, cost_optimal: Bool
) -> Int:
    return _compute_num_partitions[8, True, HALF_SMS, False](
        batch, cache, q, page, SM_COUNT, cost_optimal_split=cost_optimal
    )


def test_cost_model_split_is_pinned() raises:
    """Pin the shapes the cost model was fitted on.

    The dispatch reaches this path only for an unfolded sparse spec-decode
    verify, so `q > 1` throughout; `q = 1` is gated off at the call site and is
    deliberately not pinned here. Values are measured, not derived.
    """
    # The production shape. `cost_optimal_split` is what changes the answer.
    _expect("production shape, cost model on", _np_h8(8, 2048, 6, 64, True), 3)
    _expect(
        "production shape, cost model off", _np_h8(8, 2048, 6, 64, False), 8
    )

    # Batch curve at the production cache and q. Splitting stops paying as the
    # batch alone fills the machine.
    _expect("batch 1", _np_h8(1, 2048, 6, 64, True), 20)
    _expect("batch 2", _np_h8(2, 2048, 6, 64, True), 9)
    _expect("batch 4", _np_h8(4, 2048, 6, 64, True), 4)
    _expect("batch 16", _np_h8(16, 2048, 6, 64, True), 3)
    _expect("batch 32", _np_h8(32, 2048, 6, 64, True), 3)
    _expect("batch 128", _np_h8(128, 2048, 6, 64, True), 1)

    # q curve at the production batch: more query positions means more CTAs per
    # split, so the affordable split shrinks.
    _expect("q 2", _np_h8(8, 2048, 2, 64, True), 9)
    _expect("q 3", _np_h8(8, 2048, 3, 64, True), 4)
    _expect("q 5", _np_h8(8, 2048, 5, 64, True), 3)
    _expect("q 7", _np_h8(8, 2048, 7, 64, True), 2)
    _expect("q 8", _np_h8(8, 2048, 8, 64, True), 2)

    # The model minimises over gathered tokens, so page size does not enter it.
    _expect("page 128, cache 512", _np_h8(8, 512, 6, 128, True), 3)
    _expect("page 128, cache 2048", _np_h8(8, 2048, 6, 128, True), 3)
    _expect("page 128, cache 4096", _np_h8(8, 4096, 6, 128, True), 3)


def test_multi_head_group_ignores_the_cost_model() raises:
    """The 128-head path documents that it ignores `cost_optimal_split`. That
    claim is otherwise only prose, and the flag is threaded through the shared
    routing function.
    """
    var batches = [1, 2, 8, 16, 32, 128]
    var caches = [251, 512, 2048, 4099]
    for bi in range(len(batches)):
        for ci in range(len(caches)):
            var off = _compute_num_partitions[128, True, HALF_SMS, False](
                batches[bi],
                caches[ci],
                1,
                64,
                SM_COUNT,
                cost_optimal_split=False,
            )
            var on = _compute_num_partitions[128, True, HALF_SMS, False](
                batches[bi],
                caches[ci],
                1,
                64,
                SM_COUNT,
                cost_optimal_split=True,
            )
            if off != on:
                raise Error(
                    (
                        "multi-head-group split changed with the cost model at"
                        " batch="
                    ),
                    batches[bi],
                    " cache=",
                    caches[ci],
                    ": ",
                    off,
                    " vs ",
                    on,
                )


# Every head count `compute_mla_dispatch_scalars_runtime` enumerates.
comptime _RUNTIME_HEAD_COUNTS = [8, 12, 16, 24, 32, 48, 64, 128]


def _probe_shapes() -> List[Tuple[Int, Int, Int]]:
    """`(batch, cache, q)` triples that spread the head counts apart.

    Within one head group `num_heads` reaches the split count only through the
    spec-decode fold predicate `num_heads * q <= 64`, so every count collapses
    onto the same answer at `q = 1` and a decode-shaped probe would pin
    nothing. The last shape is the one where `is_fp8_kv` moves the np=1 clamp.
    """
    return [
        (4, 8192, 4),
        (8, 8192, 2),
        (2, 8192, 8),
        (4, 8192, 5),
        (64, 256, 1),
    ]


def _comptime_np_table[is_fp8_kv: Bool]() -> List[Int]:
    """Split count per (head count, probe shape), head-count major."""
    var shapes = _probe_shapes()
    var table = List[Int]()
    comptime for num_heads in _RUNTIME_HEAD_COUNTS:
        for si in range(len(shapes)):
            table.append(
                compute_mla_dispatch_scalars[num_heads, is_fp8_kv=is_fp8_kv](
                    shapes[si][0], shapes[si][1], shapes[si][2], SM_COUNT
                )[2]
            )
    return table^


def _head_counts() -> List[Int]:
    """`_RUNTIME_HEAD_COUNTS` as a runtime list, for the pairwise sweep."""
    var out = List[Int]()
    comptime for num_heads in _RUNTIME_HEAD_COUNTS:
        out.append(num_heads)
    return out^


def _expect_at(
    what: StringLiteral,
    num_heads: Int,
    is_fp8_kv: Bool,
    shape: Tuple[Int, Int, Int],
    got: Int,
    want: Int,
) raises:
    if got != want:
        raise Error(
            what,
            " at num_heads=",
            num_heads,
            " is_fp8_kv=",
            is_fp8_kv,
            " (batch=",
            shape[0],
            " cache=",
            shape[1],
            " q=",
            shape[2],
            "): got ",
            got,
            ", want ",
            want,
        )


def test_runtime_binding_matches_its_own_specialization() raises:
    """The runtime entry point hand-writes one arm per head count, and nothing
    else checks that an arm forwards to the specialization it names.

    The GPU tests cannot: they build `MLADispatchScalarArgs[num_heads=...]`,
    which calls `compute_mla_dispatch_scalars` at comptime and never crosses
    the runtime enumeration. A copy-paste slip binding a count to a
    neighbour's instantiation would leave them all green.
    """
    var shapes = _probe_shapes()
    comptime for is_fp8_kv in [False, True]:
        var want = _comptime_np_table[is_fp8_kv]()
        var i = 0
        comptime for num_heads in _RUNTIME_HEAD_COUNTS:
            for si in range(len(shapes)):
                var shape = shapes[si]
                var got = compute_mla_dispatch_scalars_runtime(
                    shape[0], shape[1], shape[2], num_heads, is_fp8_kv, SM_COUNT
                )
                _expect_at(
                    "batch_size", num_heads, is_fp8_kv, shape, got[0], shape[0]
                )
                _expect_at(
                    "q_max_seq_len",
                    num_heads,
                    is_fp8_kv,
                    shape,
                    got[1],
                    shape[2],
                )
                _expect_at(
                    "num_partitions",
                    num_heads,
                    is_fp8_kv,
                    shape,
                    got[2],
                    want[i],
                )
                i += 1


def _separable_in_principle(a: Int, b: Int) -> Bool:
    """Whether any input at all can tell two head counts apart.

    Across head groups `ceildiv(num_heads, 64)` scales the CTA count. Within
    one, `num_heads` reaches the split count only through the fold predicate
    `num_heads * q <= 64`, so two counts on the same side of it for every `q`
    return identical metadata for every input, forever. Beyond `q = 64` both
    products exceed 64 for any positive count, so the sweep is exhaustive.
    """
    if ceildiv(a, 64) != ceildiv(b, 64):
        return True
    for q in range(1, 65):
        if (a * q <= 64) != (b * q <= 64):
            return True
    return False


def test_the_binding_pin_is_not_vacuous() raises:
    """Shapes that map every head count onto one answer would pass the pin
    above whatever the arms are bound to.

    So hold the probe set to the separation the heuristic actually admits:
    every pair it can tell apart, these shapes must tell apart. The pairs it
    cannot -- 24/32 and 48/64, which sit on the same side of the fold predicate
    for every `q` -- must stay indistinguishable, or the reasoning in
    `_separable_in_principle` has gone stale. A mis-binding within such a pair
    returns identical metadata for every input and is unobservable here.
    """
    var shapes = _probe_shapes()
    var n = len(shapes)
    var bf16 = _comptime_np_table[False]()
    var fp8 = _comptime_np_table[True]()

    var fp8_matters = False
    for i in range(len(bf16)):
        if bf16[i] != fp8[i]:
            fp8_matters = True
    if not fp8_matters:
        raise Error(
            "no probe shape separates is_fp8_kv, so the fp8 half of the"
            " binding pin proves nothing"
        )

    var counts = _head_counts()
    for a in range(len(counts)):
        for b in range(a + 1, len(counts)):
            var separated = False
            for si in range(n):
                if (
                    bf16[a * n + si] != bf16[b * n + si]
                    or fp8[a * n + si] != fp8[b * n + si]
                ):
                    separated = True
            var expected = _separable_in_principle(counts[a], counts[b])
            if separated != expected:
                raise Error(
                    "num_heads=",
                    counts[a],
                    " vs ",
                    counts[b],
                    ": probe shapes ",
                    "separate" if separated else "do not separate",
                    " them, but the heuristic ",
                    "can" if expected else "cannot",
                )


def test_unenumerated_head_count_raises() raises:
    """The enumeration is deliberately partial, so the fall-through is a
    supported outcome and has to stay an error rather than a silent answer.
    """
    var raised = False
    try:
        _ = compute_mla_dispatch_scalars_runtime(1, 1024, 1, 7, False, SM_COUNT)
    except:
        raised = True
    if not raised:
        raise Error("unenumerated num_heads=7 returned instead of raising")


def main() raises:
    test_bucket_tables_agree()
    test_every_split_has_a_combine_kernel()
    test_cost_model_split_is_pinned()
    test_multi_head_group_ignores_the_cost_model()
    test_runtime_binding_matches_its_own_specialization()
    test_the_binding_pin_is_not_vacuous()
    test_unenumerated_head_count_raises()
    print("split policy pins OK")
