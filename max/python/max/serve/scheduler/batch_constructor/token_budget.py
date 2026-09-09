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

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from max.pipelines.context import TextContext


class RequestType(str, Enum):
    """Enumeration of high-level request types handled by the scheduler."""

    TG = "text_generation"
    CE = "context_encoding"
    MIXED = "mixed"
    UNKNOWN = "unknown"

    @classmethod
    def all(cls) -> list[RequestType]:
        """Return all known request types that participate in batching."""
        return [RequestType.TG, RequestType.CE, RequestType.MIXED]


class BudgetStatus(str, Enum):
    """Enumeration describing the result of applying a token budget to a context."""

    BUDGET_AVAILABLE = "budget_available"
    """Context fits within budget with remaining capacity for more contexts."""
    BUDGET_EXHAUSTED = "budget_exhausted"
    """Context cannot be added, even with chunking (budget full or limit reached)."""
    BUDGET_REACHED = "budget_reached"
    """Context fits (possibly after chunking) and budget is now at/near capacity."""


class TokenBudget(ABC):
    """Abstract base class for token budgets used during batch construction.

    A ``TokenBudget`` tracks how many tokens have been consumed from a particular
    dimension (for example, active prompt tokens or total context length) and
    exposes a common protocol for:

    * Checking whether a :class:`TextContext` can be admitted to the
      batch via :meth:`status_after_context`.
    * Recording the token cost once a context has been accepted via
      :meth:`add_to_budget`.

    Implementations are free to interpret the token dimension they track, but
    they must respect the following contract:

    * If ``allow_chunking`` is True, implementations **may** call
      ``context.chunk`` to reduce the effective token cost against
      the budget.
    * :meth:`add_to_budget` is only called after a non-``BUDGET_EXHAUSTED``
      status and is responsible for incrementing :attr:`used` by the same
      effective token cost that was evaluated in :meth:`status_after_context`.
    """

    def __init__(
        self,
        capacity: int,
        allow_chunking: bool,
        applicable_types: list[RequestType],
        min_chunk_tokens: int = 0,
    ) -> None:
        """Initialize a generic token budget.

        Args:
            capacity: Maximum number of tokens that may be consumed by this
                budget.
            allow_chunking: Whether this budget is permitted to shrink
                :class:`TextContext` instances via ``context.chunk`` in
                order to fit within the remaining capacity.
            applicable_types: Request types that this budget applies to. If the
                active or incoming request type is not in this list, the budget
                is effectively a no-op for that context.
            min_chunk_tokens: When > 0, a split never creates a piece
                (chunk or remainder) smaller than this: the cut moves
                earlier to protect the remainder, and contexts with no
                legal cut point are refused. 0 disables the floor.
        """
        self.capacity = capacity
        """Maximum number of tokens allowed for this budget."""
        self.allow_chunking = allow_chunking
        """Whether this budget may shrink the context via ``context.chunk`` in order to fit within the remaining capacity."""
        self.applicable_types = applicable_types
        self.min_chunk_tokens = min_chunk_tokens
        """Floor on the size of any split piece (0 = no floor)."""

        self.used = 0
        """Number of tokens currently consumed from this budget."""
        self.active_request_type = RequestType.UNKNOWN

    @property
    def remaining(self) -> int:
        """Return the remaining token capacity for this budget."""
        return self.capacity - self.used

    @abstractmethod
    def status_after_context(
        self,
        context: TextContext,
        request_type: RequestType,
    ) -> BudgetStatus:
        """Compute this budget's status after hypothetically adding a context.

        Subclasses must implement this to evaluate the effective token cost of
        ``context`` against the remaining capacity and return an appropriate
        :class:`BudgetStatus`. Implementations may mutate ``context`` (for
        example, via chunking) but must not update :attr:`used`.
        """

    @abstractmethod
    def add_to_budget(
        self,
        context: TextContext,
        request_type: RequestType,
    ) -> None:
        """Apply a previously-evaluated context's token cost to this budget.

        This method is called only after :meth:`status_after_context` has
        returned a non-:data:`BudgetStatus.BUDGET_EXHAUSTED` result for the same
        ``context`` and ``request_type``. Subclasses must update
        :attr:`used` by the same effective token cost that was evaluated in
        :meth:`status_after_context`.
        """


class TokenBudgetCollection:
    """Composite applying multiple :class:`TokenBudget` instances to a context.

    This helper allows the scheduler to treat several independent budgets
    (for example, active-token and total-context budgets) as a single logical
    budget. All budgets in the collection are evaluated for each context.
    """

    def __init__(self, token_budgets: list[TokenBudget]) -> None:
        """Create a collection of token budgets applied to the same context.

        The collection evaluates budgets in order and short-circuits on the
        first non-:data:`BudgetStatus.BUDGET_AVAILABLE` result. This allows the
        scheduler to enforce several independent limits (for example, active
        token and total-context budgets) with a single interface.

        Args:
            token_budgets: The list of budgets to apply to each context.
        """
        self.token_budgets = token_budgets

    @property
    def remaining(self) -> int:
        """Return the minimum remaining capacity across all underlying budgets.

        The collection behaves as if constrained by the most restrictive budget,
        so its remaining capacity is defined as the minimum ``remaining`` value
        among its constituent budgets.
        """
        min_val = float("inf")
        for token_budget in self.token_budgets:
            remaining = token_budget.remaining
            min_val = min(min_val, remaining)

        if min_val == float("inf"):
            raise ValueError(
                "remaining cannot be calculated without any token_budgets"
            )

        return int(min_val)

    def status_after_context(
        self,
        context: TextContext,
        request_type: RequestType,
    ) -> BudgetStatus:
        """Evaluate all budgets against a context and return the first violation.

        Budgets are evaluated in the order they were provided at construction
        time. The first budget that returns a status other than
        :data:`BudgetStatus.BUDGET_AVAILABLE` determines the overall result.
        If all budgets report :data:`BudgetStatus.BUDGET_AVAILABLE`, that
        status is returned.

        Args:
            context: The context being considered for inclusion in the batch.
            request_type: The type of request being evaluated.

        Returns:
            The first non-available :class:`BudgetStatus` reported by any
            underlying budget, or :data:`BudgetStatus.BUDGET_AVAILABLE` if
            all budgets accept the context.
        """
        for token_budget in self.token_budgets:
            status = token_budget.status_after_context(context, request_type)
            if status != BudgetStatus.BUDGET_AVAILABLE:
                return status
        return BudgetStatus.BUDGET_AVAILABLE

    def add_to_budget(
        self,
        context: TextContext,
        request_type: RequestType,
    ) -> None:
        """Apply the token cost to all underlying budgets for an accepted context.

        Args:
            context: The context that was just admitted into the batch.
            request_type: The type of request being added to the budget.
        """
        for token_budget in self.token_budgets:
            token_budget.add_to_budget(context, request_type)


class ActiveTokenBudget(TokenBudget):
    """Token budget that tracks the active window of each context.

    This budget is intended for limiting the number of tokens processed during
    a single context-encoding (CE) step. For each accepted context, the token
    cost is :attr:`TextContext.tokens.active_length`, and the budget
    may optionally shrink the active window via ``context.chunk`` when
    ``allow_chunking`` is enabled.

    The capacity and current usage are tracked via :attr:`capacity`,
    :attr:`used`, and :meth:`remaining`.
    """

    @property
    def remaining(self) -> int:
        """Return the remaining active-token capacity for this budget."""
        return self.capacity - self.used

    def status_after_context(
        self,
        context: TextContext,
        request_type: RequestType,
    ) -> BudgetStatus:
        """Evaluate whether the context's active tokens fit within the budget.

        This method examines ``context.tokens.active_length`` relative to the number of
        tokens remaining in the budget. If the active window would exceed the remaining
        capacity and ``allow_chunking`` is enabled, it may call
        ``context.tokens.chunk(tokens_remaining)`` to shrink the active window so that
        it fits.

        **Important side effects**:

        * May mutate ``context`` by reducing its active window when chunking
          is enabled.
        * Does **not** update :attr:`used`. The caller must invoke
          :meth:`add_to_budget` after a non-``BUDGET_EXHAUSTED`` status in
          order to record the cost.

        Args:
            context: The :class:`TextContext` being considered.
            request_type: The type of request being evaluated.

        Returns:
            A :class:`BudgetStatus` indicating if and how the context fits:

            * :data:`BudgetStatus.BUDGET_AVAILABLE` - context fits with room
              remaining.
            * :data:`BudgetStatus.BUDGET_REACHED` - context fits exactly or
              brings the budget to its limit.
            * :data:`BudgetStatus.BUDGET_EXHAUSTED` - context cannot be
              accommodated, even after any attempted chunking.

        Raises:
            ValueError: If chunking is enabled but ``context.chunk`` is
                unable to reduce the active window to within the remaining
                capacity.
        """
        if (
            self.active_request_type not in self.applicable_types
            and request_type not in self.applicable_types
        ):
            return BudgetStatus.BUDGET_AVAILABLE

        tokens_remaining = self.remaining

        # Already at or beyond capacity - no more contexts can be accepted.
        if tokens_remaining <= 0:
            return BudgetStatus.BUDGET_EXHAUSTED

        # Fits without any modification.
        if context.tokens.active_length == tokens_remaining:
            return BudgetStatus.BUDGET_REACHED
        elif context.tokens.active_length < tokens_remaining:
            return BudgetStatus.BUDGET_AVAILABLE

        # Would exceed the remaining capacity.
        # This is a soft limit, so we return BUDGET_REACHED instead of BUDGET_EXHAUSTED
        if not self.allow_chunking:
            return BudgetStatus.BUDGET_REACHED

        # Try to shrink the active window so that it fits. The min-chunk
        # floor moves the cut earlier to protect the remainder, or refuses
        # the split when no legal cut point exists.
        chunk_size = tokens_remaining
        if self.min_chunk_tokens > 0:
            active_length = context.tokens.active_length
            if active_length - chunk_size < self.min_chunk_tokens:
                chunk_size = active_length - self.min_chunk_tokens
            if chunk_size < self.min_chunk_tokens:
                return BudgetStatus.BUDGET_EXHAUSTED
        try:
            context.tokens.chunk(chunk_size)
            return BudgetStatus.BUDGET_REACHED
        except ValueError:
            return BudgetStatus.BUDGET_EXHAUSTED

    def add_to_budget(
        self,
        context: TextContext,
        request_type: RequestType,
    ) -> None:
        """Record the token cost for an accepted context's active tokens.

        This should be called only after :meth:`status_after_context` has
        returned a non-:data:`BudgetStatus.BUDGET_EXHAUSTED` result for the
        same ``context``.

        Args:
            context: The context that was just admitted into the batch (possibly
                after being chunked).
            request_type: The type of request being added to the budget.
        """
        if (
            self.active_request_type != RequestType.MIXED
            and request_type != self.active_request_type
        ):
            self.active_request_type = request_type

        self.used += context.tokens.active_length


class TotalContextTokenBudget(TokenBudget):
    """Token budget that tracks the full context length for each request.

    Unlike :class:`ActiveTokenBudget`, which only costs the active window per
    step, this budget has an effective cost derived from
    :attr:`len(TextContext.tokens)` and the planned number of
    generation steps. It is intended for enforcing limits such as
    ``max_batch_total_tokens`` that bound the total number of tokens resident
    in a batch across multiple steps. The effective cost is rounded up to the
    configured ``page_size`` so per-request lengths match PagedKVCache
    alignment.
    """

    def __init__(
        self,
        capacity: int,
        allow_chunking: bool,
        applicable_types: list[RequestType],
        cost_alignment: int = 1,
    ) -> None:
        """Initialize a total-context token budget.

        Args:
            capacity: Maximum number of tokens that may be consumed by this
                budget.
            allow_chunking: Whether this budget is permitted to shrink
                :class:`TextContext` instances via ``context.chunk`` in
                order to fit within the remaining capacity.
            applicable_types: Request types that this budget applies to.
            cost_alignment: Alignment used to align effective costs for this budget.
                Defaults to 1, which means no alignment is applied.
        """
        super().__init__(
            capacity=capacity,
            allow_chunking=allow_chunking,
            applicable_types=applicable_types,
        )

        self.cost_alignment = cost_alignment

    @staticmethod
    def align_up(x: int, align: int) -> int:
        """Round x up to the nearest multiple of align."""
        return (x + align - 1) // align * align

    @property
    def remaining(self) -> int:
        """Return the remaining total-context capacity for this budget."""
        return self.capacity - self.used

    def status_after_context(
        self,
        context: TextContext,
        request_type: RequestType,
    ) -> BudgetStatus:
        """Evaluate whether the context's total length fits within the budget.

        This method considers an effective cost based on
        :attr:`len(TextContext.tokens)` against the remaining capacity. The
        effective cost is rounded up to the page size so per-request lengths
        match the PagedKVCache alignment that some GPU kernels expect.

        Args:
            context: The :class:`TextContext` being considered.
            request_type: The type of request being evaluated.

        Returns:
            A :class:`BudgetStatus` indicating if and how the context fits:

            * :data:`BudgetStatus.BUDGET_AVAILABLE` - context fits with room
              remaining.
            * :data:`BudgetStatus.BUDGET_REACHED` - context exactly consumes
              the remaining capacity.
            * :data:`BudgetStatus.BUDGET_EXHAUSTED` - context cannot be
              accommodated within the remaining capacity.
        """

        if request_type not in self.applicable_types:
            return BudgetStatus.BUDGET_AVAILABLE

        tokens_remaining = self.remaining

        # Already at or beyond capacity - no more contexts can be accepted.
        if tokens_remaining <= 0:
            return BudgetStatus.BUDGET_EXHAUSTED

        # Match PagedKVCache page alignment for GPU kernels.
        total_length = self.align_up(len(context.tokens), self.cost_alignment)

        if total_length < tokens_remaining:
            return BudgetStatus.BUDGET_AVAILABLE
        elif total_length == tokens_remaining:
            return BudgetStatus.BUDGET_REACHED

        # NOTE: Skip chunking since chunking only reduces the active window and
        # cannot lower current_length, so it cannot make a total-context overage
        # fit this budget.
        return BudgetStatus.BUDGET_EXHAUSTED

    def add_to_budget(
        self,
        context: TextContext,
        request_type: RequestType,
    ) -> None:
        """Record the token cost for an accepted context's total length.

        This should be called only after :meth:`status_after_context` has
        returned a non-:data:`BudgetStatus.BUDGET_EXHAUSTED` result for the
        same ``context``.

        **Side effect**:
            Increments :attr:`used` by the page-aligned ``len(context.tokens)``.

        Args:
            context: The context that was just admitted into the batch.
            request_type: The type of request being added to the budget.
        """
        self.used += self.align_up(len(context.tokens), self.cost_alignment)
