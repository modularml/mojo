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

import numpy as np
from max.pipelines.context import TextContext, TokenBuffer
from max.serve.scheduler.batch_constructor.token_budget import (
    ActiveTokenBudget,
    BudgetStatus,
    RequestType,
    TotalContextTokenBudget,
)


def test_token_budget__active_token_budget_with_chunking() -> None:
    # We have a budget of 100 tokens, and we are allowing chunking
    active_token_budget = ActiveTokenBudget(
        capacity=100,
        allow_chunking=True,
        applicable_types=[RequestType.CE, RequestType.TG, RequestType.MIXED],
    )

    for i in range(11):
        context = TextContext(
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)), max_length=100
        )

        status = active_token_budget.status_after_context(
            context, request_type=RequestType.CE
        )

        if i < 9:
            assert status == BudgetStatus.BUDGET_AVAILABLE
            active_token_budget.add_to_budget(
                context, request_type=RequestType.CE
            )
        elif i == 9:
            # The tenth context should hit the budget limit exactly
            assert status == BudgetStatus.BUDGET_REACHED
            active_token_budget.add_to_budget(
                context, request_type=RequestType.CE
            )
        else:
            # There is no room left in the budget
            assert status == BudgetStatus.BUDGET_EXHAUSTED

    assert active_token_budget.remaining == 0


def test_token_budget__active_token_budget_without_chunking() -> None:
    active_token_budget = ActiveTokenBudget(
        capacity=100, allow_chunking=False, applicable_types=[RequestType.CE]
    )

    for i in range(11):
        context = TextContext(
            tokens=TokenBuffer(np.ones(11, dtype=np.int64)), max_length=100
        )

        status = active_token_budget.status_after_context(
            context, request_type=RequestType.CE
        )

        if i < 9:
            assert status == BudgetStatus.BUDGET_AVAILABLE
            active_token_budget.add_to_budget(
                context, request_type=RequestType.CE
            )
        elif i == 9:
            assert status == BudgetStatus.BUDGET_REACHED
            active_token_budget.add_to_budget(
                context, request_type=RequestType.CE
            )
        else:
            assert status == BudgetStatus.BUDGET_EXHAUSTED

    # This is a soft limit, so we should be able to go over the budget by a few tokens.
    assert active_token_budget.remaining < 0


def test_token_budget__min_chunk_floor_moves_cut_to_protect_remainder() -> None:
    # A boundary cut at 100 tokens would leave an 8-token remainder; with a
    # 32-token floor the cut moves earlier so the remainder stays legal.
    budget = ActiveTokenBudget(
        capacity=100,
        allow_chunking=True,
        applicable_types=[RequestType.CE],
        min_chunk_tokens=32,
    )
    context = TextContext(
        tokens=TokenBuffer(np.ones(108, dtype=np.int64)), max_length=200
    )
    status = budget.status_after_context(context, request_type=RequestType.CE)
    assert status == BudgetStatus.BUDGET_REACHED
    assert context.tokens.active_length == 108 - 32


def test_token_budget__min_chunk_floor_refuses_degenerate_splits() -> None:
    budget = ActiveTokenBudget(
        capacity=40,
        allow_chunking=True,
        applicable_types=[RequestType.CE],
        min_chunk_tokens=32,
    )

    # Cutting 50 tokens at the 40-token boundary leaves a 10-token remainder;
    # moving the cut to protect it (50 - 32 = 18) would make the chunk itself
    # sub-floor. No legal cut point exists, so the split is refused and the
    # context is left untouched for a later, roomier step.
    context = TextContext(
        tokens=TokenBuffer(np.ones(50, dtype=np.int64)), max_length=200
    )
    status = budget.status_after_context(context, request_type=RequestType.CE)
    assert status == BudgetStatus.BUDGET_EXHAUSTED
    assert context.tokens.active_length == 50

    # 80 tokens splits legally at the boundary: both pieces are 40 >= 32.
    context = TextContext(
        tokens=TokenBuffer(np.ones(80, dtype=np.int64)), max_length=200
    )
    status = budget.status_after_context(context, request_type=RequestType.CE)
    assert status == BudgetStatus.BUDGET_REACHED
    assert context.tokens.active_length == 40


def test_token_budget__min_chunk_floor_disabled_cuts_at_boundary() -> None:
    # Default floor of 0 preserves the existing behavior: cuts land exactly
    # on the budget boundary even when that leaves a tiny remainder.
    budget = ActiveTokenBudget(
        capacity=100,
        allow_chunking=True,
        applicable_types=[RequestType.CE],
    )
    context = TextContext(
        tokens=TokenBuffer(np.ones(108, dtype=np.int64)), max_length=200
    )
    status = budget.status_after_context(context, request_type=RequestType.CE)
    assert status == BudgetStatus.BUDGET_REACHED
    assert context.tokens.active_length == 100


def test_token_budget__total_context_budget_with_cost_alignment_alignment() -> (
    None
):
    """TotalContextTokenBudget should align the effective cost to the page size."""

    total_context_budget = TotalContextTokenBudget(
        capacity=30,
        allow_chunking=False,
        applicable_types=[RequestType.CE],
        cost_alignment=8,  # token cost is aligned to 8 for this test
    )

    # current_length = 10 => aligned cost = align_up(10, 8) = 16
    context = TextContext(
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)), max_length=100
    )
    status = total_context_budget.status_after_context(
        context, request_type=RequestType.CE
    )
    assert status == BudgetStatus.BUDGET_AVAILABLE

    # Commit the aligned length to the budget:
    # used = 16, remaining = 30 - 16 = 14.
    total_context_budget.add_to_budget(context, request_type=RequestType.CE)
    assert total_context_budget.remaining == 14

    # Now with remaining=14, a new 10-token context gives
    # aligned cost = align_up(10, 8) = 16
    # which is greater than remaining=14, so it should be rejected.
    context2 = TextContext(
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)), max_length=100
    )
    status2 = total_context_budget.status_after_context(
        context2, request_type=RequestType.CE
    )
    assert status2 == BudgetStatus.BUDGET_EXHAUSTED


def test_token_budget__total_context_budget_available_and_reached() -> None:
    """TotalContextTokenBudget should check capacity against context length."""
    total_budget = TotalContextTokenBudget(
        capacity=25,
        allow_chunking=False,
        applicable_types=[RequestType.CE, RequestType.TG, RequestType.MIXED],
        cost_alignment=1,
    )

    # current_length = 10 < 25: fits
    context = TextContext(
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)), max_length=100
    )
    status = total_budget.status_after_context(
        context, request_type=RequestType.CE
    )
    assert status == BudgetStatus.BUDGET_AVAILABLE

    # Commit 10 tokens: remaining = 25 - 10 = 15.
    total_budget.add_to_budget(context, request_type=RequestType.CE)
    assert total_budget.remaining == 15

    # A second 15-token context exactly reaches the budget.
    context2 = TextContext(
        tokens=TokenBuffer(np.ones(15, dtype=np.int64)), max_length=100
    )
    status2 = total_budget.status_after_context(
        context2, request_type=RequestType.CE
    )
    assert status2 == BudgetStatus.BUDGET_REACHED


def test_token_budget__total_context_budget_exhausted() -> None:
    """TotalContextTokenBudget should reject contexts that would overflow."""
    total_budget = TotalContextTokenBudget(
        capacity=14,
        allow_chunking=False,
        applicable_types=[RequestType.CE],
        cost_alignment=1,
    )

    # current_length = 15 > 14: exceeds capacity
    context = TextContext(
        tokens=TokenBuffer(np.ones(15, dtype=np.int64)), max_length=100
    )
    status = total_budget.status_after_context(
        context, request_type=RequestType.CE
    )
    assert status == BudgetStatus.BUDGET_EXHAUSTED


def test_token_budget__total_context_budget_with_chunking_exhausts_overage() -> (
    None
):
    """TotalContextTokenBudget should not chunk; total-context overage is exhausted."""
    total_budget = TotalContextTokenBudget(
        capacity=20,
        allow_chunking=True,
        applicable_types=[RequestType.CE],
        cost_alignment=1,  # no alignment needed for this test
    )

    context = TextContext(
        tokens=TokenBuffer(np.ones(30, dtype=np.int64)), max_length=100
    )
    assert len(context.tokens) == 30
    assert context.tokens.active_length == 30

    status = total_budget.status_after_context(
        context, request_type=RequestType.CE
    )
    assert status == BudgetStatus.BUDGET_EXHAUSTED
    assert len(context.tokens) == 30
    assert context.tokens.active_length == 30
    assert total_budget.used == 0
    assert total_budget.remaining == 20


def test_token_budget__total_context_budget_overage_does_not_chunk() -> None:
    """TotalContextTokenBudget should not chunk even when remaining exceeds active_length."""
    total_budget = TotalContextTokenBudget(
        capacity=100,
        allow_chunking=True,
        applicable_types=[RequestType.CE],
        cost_alignment=1,  # no alignment needed for this test
    )

    context = TextContext(
        tokens=TokenBuffer(np.ones(200, dtype=np.int64)), max_length=300
    )
    context.tokens.skip_processing(190)

    assert len(context.tokens) == 200
    assert context.tokens.active_length == 10

    status = total_budget.status_after_context(
        context, request_type=RequestType.CE
    )
    assert status == BudgetStatus.BUDGET_EXHAUSTED


def test_token_budget__total_context_budget_chunking_disabled_for_unit_active_length() -> (
    None
):
    """Chunking is not applied when active_length == 1, even if allow_chunking is True."""
    total_budget = TotalContextTokenBudget(
        capacity=10,
        allow_chunking=True,
        applicable_types=[RequestType.CE],
        cost_alignment=1,  # no alignment needed for this test
    )

    # Create a context where only a single token is active, but the total
    # sequence length is much larger (simulating TG-style usage).
    context = TextContext(
        tokens=TokenBuffer(np.ones(50, dtype=np.int64)), max_length=100
    )
    context.tokens.skip_processing(49)

    assert context.tokens.active_length == 1
    assert len(context.tokens) == 50

    status = total_budget.status_after_context(
        context, request_type=RequestType.CE
    )
    # Since active_length == 1, TotalContextTokenBudget should not attempt
    # chunking and must report the budget as exhausted.
    assert status == BudgetStatus.BUDGET_EXHAUSTED
    assert total_budget.used == 0
    assert total_budget.remaining == 10


def test_token_budget__total_context_budget__ce_after_tg() -> None:
    # This scenario, would only happen with mixed batching
    total_budget = TotalContextTokenBudget(
        capacity=10,
        allow_chunking=True,
        applicable_types=[RequestType.MIXED, RequestType.CE],
        cost_alignment=1,  # no alignment needed for this test
    )

    # Create
    context = TextContext(
        tokens=TokenBuffer(np.ones(20, dtype=np.int64)), max_length=100
    )
    # Move it to the end, assuming it is a TG request
    context.tokens.skip_processing(19)

    # This should still show as available, as its a TG request, and this budget does not apply
    status = total_budget.status_after_context(
        context, request_type=RequestType.TG
    )
    assert status == BudgetStatus.BUDGET_AVAILABLE

    # Add to budget
    total_budget.add_to_budget(context, request_type=RequestType.TG)

    # Create a new CE request with a small number of tokens
    context = TextContext(
        tokens=TokenBuffer(np.ones(3, dtype=np.int64)), max_length=100
    )

    # This should be rejected, as we already have a TG object in batch that is beyond the threshold.
    status = total_budget.status_after_context(
        context, request_type=RequestType.CE
    )
    assert status == BudgetStatus.BUDGET_EXHAUSTED
