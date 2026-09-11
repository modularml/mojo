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

from std.testing import (
    assert_equal,
    assert_true,
    assert_false,
    TestSuite,
)
from std.testing.prop._shrinking._passes import (
    DeleteChunkPass,
    ReduceChunkPass,
    ZeroChunkPass,
    binary_search_value,
    delete_chunk,
)
from std.testing.prop._shrinking._shrinker import (
    ShrinkOracle,
    Stream,
    stream_is_better,
)


struct FakeOracle[is_interesting: def(Stream) thin -> Bool](ShrinkOracle):
    """Test oracle backed by a programmable interesting-ness predicate.

    Drives shrink passes in unit tests without requiring a `Strategy`,
    `Rng`, captured callable, or expected `Error`.
    """

    var best: Stream
    var calls: Int

    def __init__(out self, var best: Stream):
        self.best = best^
        self.calls = 0

    def try_stream(mut self, stream: Stream) -> Bool:
        self.calls += 1
        if Self.is_interesting(stream) and stream_is_better(stream, self.best):
            self.best = stream.copy()
            return True
        return False

    def best_stream[o: ImmOrigin, //](ref[o] self) -> ref[o._subtree] Stream:
        return self.best


# ===-------------------------------------------------------------------=== #
# delete_chunk helper
# ===-------------------------------------------------------------------=== #


def test_delete_chunk_removes_middle_slice() raises:
    var s: Stream = [1, 2, 3, 4, 5]
    var out = delete_chunk(s, 1, 2)
    assert_equal(len(out), 3)
    assert_equal(out[0], 1)
    assert_equal(out[1], 4)
    assert_equal(out[2], 5)


def test_delete_chunk_removes_prefix() raises:
    var s: Stream = [1, 2, 3, 4]
    var out = delete_chunk(s, 0, 2)
    assert_equal(len(out), 2)
    assert_equal(out[0], 3)
    assert_equal(out[1], 4)


def test_delete_chunk_removes_suffix() raises:
    var s: Stream = [1, 2, 3, 4]
    var out = delete_chunk(s, 2, 2)
    assert_equal(len(out), 2)
    assert_equal(out[0], 1)
    assert_equal(out[1], 2)


# ===-------------------------------------------------------------------=== #
# delete_chunk_pass
# ===-------------------------------------------------------------------=== #


def test_delete_chunk_pass_drives_to_minimum_length() raises:
    def has_any(s: Stream) -> Bool:
        return len(s) > 0

    var stream: Stream = [5, 7, 11, 3]
    var oracle = FakeOracle[has_any](stream^)
    var pass_ = DeleteChunkPass()

    var improved = True
    while improved:
        improved = pass_.run(oracle)

    assert_equal(len(oracle.best), 1)


def test_delete_chunk_pass_returns_false_when_already_minimal() raises:
    def has_any(s: Stream) -> Bool:
        return len(s) > 0

    var stream: Stream = [42]
    var oracle = FakeOracle[has_any](stream^)
    var pass_ = DeleteChunkPass()
    assert_false(pass_.run(oracle))
    assert_equal(len(oracle.best), 1)
    assert_equal(oracle.best[0], 42)


def test_delete_chunk_pass_preserves_required_elements() raises:
    # Predicate requires the value 99 to be present somewhere.

    def has_99(s: Stream) -> Bool:
        for v in s:
            if v == 99:
                return True
        return False

    var stream: Stream = [1, 2, 99, 3, 4]
    var oracle = FakeOracle[has_99](stream^)
    var pass_ = DeleteChunkPass()

    var improved = True
    while improved:
        improved = pass_.run(oracle)

    # The 99 must survive; nothing else is required.
    var found = False
    for v in oracle.best:
        if v == 99:
            found = True
    assert_true(found)
    assert_equal(len(oracle.best), 1)


# ===-------------------------------------------------------------------=== #
# zero_chunk_pass
# ===-------------------------------------------------------------------=== #


def test_zero_chunk_pass_zeros_when_predicate_allows() raises:
    # Predicate accepts any stream with at least one nonzero element.

    def has_nonzero(s: Stream) -> Bool:
        for v in s:
            if v != 0:
                return True
        return False

    var stream: Stream = [1, 2, 3, 4, 5, 6, 7, 8]
    var oracle = FakeOracle[has_nonzero](stream^)
    var pass_ = ZeroChunkPass()

    var improved = True
    while improved:
        improved = pass_.run(oracle)

    var nonzero = 0
    for v in oracle.best:
        if v != 0:
            nonzero += 1
    # Exactly one element must remain nonzero — anything more zeroed would
    # violate the predicate.
    assert_equal(nonzero, 1)


def test_zero_chunk_pass_no_op_when_zeroing_breaks_predicate() raises:
    # Predicate requires *every* element to be nonzero — zeroing breaks it.

    def all_nonzero(s: Stream) -> Bool:
        if len(s) == 0:
            return False
        for v in s:
            if v == 0:
                return False
        return True

    var stream: Stream = [7, 7, 7, 7]
    var oracle = FakeOracle[all_nonzero](stream^)
    var pass_ = ZeroChunkPass()
    assert_false(pass_.run(oracle))
    for v in oracle.best:
        assert_equal(v, 7)


# ===-------------------------------------------------------------------=== #
# reduce_chunk_pass
# ===-------------------------------------------------------------------=== #


def test_reduce_chunk_pass_drives_value_to_threshold() raises:
    def first_at_least_5(s: Stream) -> Bool:
        return len(s) > 0 and s[0] >= 5

    var stream: Stream = [100]
    var oracle = FakeOracle[first_at_least_5](stream^)
    var pass_ = ReduceChunkPass()
    assert_true(pass_.run(oracle))
    assert_equal(oracle.best[0], 5)


def test_reduce_chunk_pass_returns_false_when_already_minimal() raises:
    def first_at_least_5(s: Stream) -> Bool:
        return len(s) > 0 and s[0] >= 5

    var stream: Stream = [5]
    var oracle = FakeOracle[first_at_least_5](stream^)
    var pass_ = ReduceChunkPass()
    assert_false(pass_.run(oracle))
    assert_equal(oracle.best[0], 5)


# ===-------------------------------------------------------------------=== #
# binary_search_value
# ===-------------------------------------------------------------------=== #


def test_binary_search_value_converges_to_threshold() raises:
    def first_at_least_7(s: Stream) -> Bool:
        return len(s) > 0 and s[0] >= 7

    var stream: Stream = [100]
    var oracle = FakeOracle[first_at_least_7](stream^)
    var improved = False
    binary_search_value(oracle, 0, 100, improved)
    assert_true(improved)
    assert_equal(oracle.best[0], 7)


def test_binary_search_value_no_op_when_no_smaller_works() raises:
    # Only the exact starting value is interesting — no smaller value works.

    def exactly_42(s: Stream) -> Bool:
        return len(s) > 0 and s[0] == 42

    var stream: Stream = [42]
    var oracle = FakeOracle[exactly_42](stream^)
    var improved = False
    binary_search_value(oracle, 0, 42, improved)
    assert_false(improved)
    assert_equal(oracle.best[0], 42)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
