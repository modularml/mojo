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

"""Unit tests for benchmark_shared.datasets.image_augmentation."""

from __future__ import annotations

import logging

import pytest
from max.benchmark.benchmark_shared.datasets.chat_judge import (
    ChatJudgeChatSamples,
)
from max.benchmark.benchmark_shared.datasets.distribution import (
    BaseDistribution,
)
from max.benchmark.benchmark_shared.datasets.image_augmentation import (
    ImageDimensions,
    _sends_no_measured_turns,
    augment_samples_with_images,
    estimate_image_token_len,
    sample_image_dimensions,
)
from max.benchmark.benchmark_shared.datasets.types import (
    ChatMessage,
    ChatSamples,
    ChatSession,
    RequestSamples,
    SampledRequest,
    SessionMessage,
)


def _dist(param: str | float) -> BaseDistribution:
    dist = BaseDistribution.from_distribution_parameter(param)
    assert dist is not None
    return dist


def test_sample_image_dimensions_square() -> None:
    width, height = sample_image_dimensions(_dist(512), _dist(1.0))
    assert width == 512
    assert height == 512


def test_sample_image_dimensions_landscape() -> None:
    width, height = sample_image_dimensions(_dist(512), _dist(2.0))
    assert width == 512
    assert height == 256


def test_sample_image_dimensions_portrait() -> None:
    width, height = sample_image_dimensions(_dist(512), _dist(0.5))
    assert height == 512
    assert width == 256


def test_sample_image_dimensions_returns_named_fields() -> None:
    result = sample_image_dimensions(_dist(512), _dist(2.0))
    assert isinstance(result, ImageDimensions)
    assert result.width == 512
    assert result.height == 256


def test_estimate_image_token_len_reference_point() -> None:
    """512x512 is the anchor: ~256 tokens, matching the prior hardcoded value."""
    assert estimate_image_token_len(512, 512) == 256


def test_estimate_image_token_len_scales_with_area() -> None:
    small = estimate_image_token_len(256, 256)
    large = estimate_image_token_len(1024, 1024)
    assert small < estimate_image_token_len(512, 512) < large


def _make_request(prompt_len: int = 100) -> SampledRequest:
    return SampledRequest(
        prompt_formatted="hello",
        prompt_len=prompt_len,
        output_len=10,
        encoded_images=[],
        ignore_eos=False,
    )


def test_estimate_image_token_len_uses_pixel_area() -> None:
    """Aspect ratio must affect cost: 1024x256 has the same area as 512x512."""
    assert estimate_image_token_len(1024, 256) == estimate_image_token_len(
        512, 512
    )
    assert estimate_image_token_len(1024, 256) < estimate_image_token_len(
        1024, 1024
    )


def test_sample_image_dimensions_rejects_non_positive_ratio() -> None:
    """Clamping used to turn a bad ratio into a 1px-wide image."""
    with pytest.raises(ValueError, match="image-aspect-ratio"):
        sample_image_dimensions(_dist(512), _dist(0.0))


def test_augment_request_samples_zero_fraction_is_noop() -> None:
    samples = RequestSamples(requests=[_make_request() for _ in range(20)])
    augment_samples_with_images(
        samples,
        image_fraction=0.0,
        image_count=1,
        image_long_side=512,
        image_aspect_ratio=1.0,
    )
    assert all(len(r.encoded_images) == 0 for r in samples.requests)


def test_augment_rejects_out_of_range_fraction() -> None:
    """A negative fraction must raise, not silently no-op, so the range check
    has to run ahead of the zero early-return or it is unreachable."""
    samples = RequestSamples(requests=[_make_request()])
    for bad in (-0.5, 1.5):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            augment_samples_with_images(
                samples,
                image_fraction=bad,
                image_count=1,
                image_long_side=512,
                image_aspect_ratio=1.0,
            )


def test_augment_request_samples_full_fraction_adds_images_to_all() -> None:
    samples = RequestSamples(requests=[_make_request() for _ in range(20)])
    augment_samples_with_images(
        samples,
        image_fraction=1.0,
        image_count="DU(2,2)",
        image_long_side=512,
        image_aspect_ratio=1.0,
    )
    for request in samples.requests:
        assert len(request.encoded_images) == 2
        # Each image should have bumped prompt_len above the original 100.
        assert request.prompt_len > 100


def test_augment_request_samples_partial_fraction_converges() -> None:
    n = 2000
    samples = RequestSamples(requests=[_make_request() for _ in range(n)])
    augment_samples_with_images(
        samples,
        image_fraction=0.3,
        image_count=1,
        image_long_side=512,
        image_aspect_ratio=1.0,
    )
    with_images = sum(1 for r in samples.requests if r.encoded_images)
    # Loose bound: a Bernoulli(0.3) draw over 2000 trials essentially never
    # lands outside +/- 0.1 of the target fraction.
    assert 0.2 * n < with_images < 0.4 * n


def _make_session(session_id: int, num_user_turns: int = 3) -> ChatSession:
    messages: list[SessionMessage] = []
    for i in range(num_user_turns):
        messages.append(
            SessionMessage(source="user", content=f"turn {i}", num_tokens=10)
        )
        messages.append(
            SessionMessage(source="assistant", content="", num_tokens=5)
        )
    return ChatSession(id=session_id, messages=messages)


def test_augment_chat_samples_first_turn_only() -> None:
    samples = ChatSamples(chat_sessions=[_make_session(i) for i in range(10)])
    augment_samples_with_images(
        samples,
        image_fraction=1.0,
        image_count=1,
        image_long_side=512,
        image_aspect_ratio=1.0,
        image_turn="first",
    )
    for session in samples.chat_sessions:
        user_messages = [m for m in session.messages if m.source == "user"]
        assert len(user_messages[0].images) == 1
        for later in user_messages[1:]:
            assert len(later.images) == 0


def test_augment_chat_samples_last_turn_only() -> None:
    samples = ChatSamples(chat_sessions=[_make_session(i) for i in range(10)])
    augment_samples_with_images(
        samples,
        image_fraction=1.0,
        image_count=1,
        image_long_side=512,
        image_aspect_ratio=1.0,
        image_turn="last",
    )
    for session in samples.chat_sessions:
        user_messages = [m for m in session.messages if m.source == "user"]
        assert len(user_messages[-1].images) == 1
        for earlier in user_messages[:-1]:
            assert len(earlier.images) == 0


def test_augment_chat_samples_every_turn() -> None:
    samples = ChatSamples(chat_sessions=[_make_session(i) for i in range(5)])
    augment_samples_with_images(
        samples,
        image_fraction=1.0,
        image_count=1,
        image_long_side=512,
        image_aspect_ratio=1.0,
        image_turn="every",
    )
    for session in samples.chat_sessions:
        for msg in session.messages:
            if msg.source == "user":
                assert len(msg.images) == 1
            else:
                assert len(msg.images) == 0


def test_augment_chat_samples_skips_unmeasurable_sessions() -> None:
    """A session whose images would push it past max_chat_len sends no
    measured turns, so it must come back untouched -- attaching anyway would
    leave num_tokens and the --dry-run tables describing an unsent request."""
    samples = ChatSamples(chat_sessions=[_make_session(0)])
    before = samples.chat_sessions[0].messages[0].num_tokens

    augment_samples_with_images(
        samples,
        image_fraction=1.0,
        image_count=1,
        image_long_side=512,
        image_aspect_ratio=1.0,
        image_turn="first",
        max_chat_len=1,
    )

    session = samples.chat_sessions[0]
    assert session.messages[0].images == []
    assert session.messages[0].num_tokens == before


def test_augment_chat_samples_counts_sessions_that_still_fit() -> None:
    samples = ChatSamples(chat_sessions=[_make_session(0)])
    augment_samples_with_images(
        samples,
        image_fraction=1.0,
        image_count=1,
        image_long_side=512,
        image_aspect_ratio=1.0,
        image_turn="first",
        max_chat_len=10**6,
    )
    assert len(samples.chat_sessions[0].messages[0].images) == 1


def test_sends_no_measured_turns_accounts_for_warmup_prefix() -> None:
    """With --warmup-to-steady-state the first measured turn is at
    prefix_turns * 2 and carries the whole prefix, so a budget check that
    only looks at messages[0] and [1] passes a session the driver abandons."""
    session = _make_session(0, num_user_turns=4)
    session.prefix_turns = 3
    # Each prefix turn costs 15; three of them exhaust a budget of 40 before
    # any measured turn is reached, even though messages[0] + [1] is only 15.
    assert session.messages[0].num_tokens + session.messages[1].num_tokens < 40
    assert _sends_no_measured_turns(session, max_chat_len=40)


def test_sends_no_measured_turns_allows_prefix_that_fits() -> None:
    session = _make_session(0, num_user_turns=4)
    session.prefix_turns = 1
    assert not _sends_no_measured_turns(session, max_chat_len=10**6)


def test_sends_no_measured_turns_charges_run_prefix_to_first_turn() -> None:
    """--force-unique-runs prepends a per-run prefix to the first measured
    turn, so a session sitting just under the budget without it is abandoned
    with it."""
    session = _make_session(0, num_user_turns=2)
    first_turn = session.messages[0].num_tokens + session.messages[1].num_tokens
    assert not _sends_no_measured_turns(session, max_chat_len=first_turn)
    assert _sends_no_measured_turns(
        session, max_chat_len=first_turn, run_prefix_len=1
    )


def test_sends_no_measured_turns_ignores_run_prefix_behind_warmup() -> None:
    """The driver only charges the run prefix at content_idx 0, so a warmup
    prefix ahead of the first measured turn means it never applies."""
    session = _make_session(0, num_user_turns=4)
    session.prefix_turns = 1
    assert not _sends_no_measured_turns(
        session, max_chat_len=10**6, run_prefix_len=10**6
    )


def test_augment_chat_samples_excludes_warmup_session_over_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The end-to-end path, not just the helper: a prefixed session the driver
    will abandon must be reported as unmeasurable, not counted as augmented."""
    session = _make_session(0, num_user_turns=4)
    session.prefix_turns = 3
    samples = ChatSamples(chat_sessions=[session])

    with caplog.at_level(logging.INFO):
        augment_samples_with_images(
            samples,
            image_fraction=1.0,
            image_count=1,
            image_long_side=512,
            image_aspect_ratio=1.0,
            image_turn="first",
            max_chat_len=40,
        )

    assert "before their first measured turn" in caplog.text
    assert "added images to 0/1 chat sessions" in caplog.text
    # Nothing counted means nothing attached.
    assert samples.chat_sessions[0].messages[0].images == []


def test_augment_request_samples_skips_prompt_with_no_user_message() -> None:
    """The driver attaches images to the first user message; with none, they
    are dropped at send time, so their tokens must never be counted."""
    system_only = SampledRequest(
        prompt_formatted=[ChatMessage(role="system", content="be terse")],
        prompt_len=5,
        output_len=8,
        encoded_images=[],
        ignore_eos=False,
    )
    samples = RequestSamples(requests=[system_only])

    augment_samples_with_images(
        samples,
        image_fraction=1.0,
        image_count=1,
        image_long_side=512,
        image_aspect_ratio=1.0,
    )

    assert system_only.encoded_images == []
    assert system_only.prompt_len == 5


def test_augment_request_samples_augments_prompt_with_a_user_message() -> None:
    with_user = SampledRequest(
        prompt_formatted=[
            ChatMessage(role="system", content="be terse"),
            ChatMessage(role="user", content="hello"),
        ],
        prompt_len=5,
        output_len=8,
        encoded_images=[],
        ignore_eos=False,
    )
    samples = RequestSamples(requests=[with_user])

    augment_samples_with_images(
        samples,
        image_fraction=1.0,
        image_count=1,
        image_long_side=512,
        image_aspect_ratio=1.0,
    )

    assert len(with_user.encoded_images) == 1
    assert with_user.prompt_len > 5


def test_augment_request_samples_augments_plain_string_prompt() -> None:
    """A bare string becomes a single user message, so it always accepts."""
    plain = SampledRequest(
        prompt_formatted="hello",
        prompt_len=5,
        output_len=8,
        encoded_images=[],
        ignore_eos=False,
    )
    samples = RequestSamples(requests=[plain])

    augment_samples_with_images(
        samples,
        image_fraction=1.0,
        image_count=1,
        image_long_side=512,
        image_aspect_ratio=1.0,
    )

    assert len(plain.encoded_images) == 1


def test_augment_samples_skips_chat_judge() -> None:
    """chat_judge_session_driver sends text-only messages, so augmenting a
    chat-judge workload would report images the server never receives."""
    samples = ChatJudgeChatSamples(
        chat_sessions=[_make_session(i) for i in range(5)]
    )
    before = [
        [m.num_tokens for m in session.messages]
        for session in samples.chat_sessions
    ]

    augment_samples_with_images(
        samples,
        image_fraction=1.0,
        image_count=1,
        image_long_side=512,
        image_aspect_ratio=1.0,
        image_turn="every",
    )

    for session in samples.chat_sessions:
        for message in session.messages:
            assert message.images == []
    after = [
        [m.num_tokens for m in session.messages]
        for session in samples.chat_sessions
    ]
    assert after == before


def test_augment_samples_invalid_fraction_raises() -> None:
    samples = RequestSamples(requests=[_make_request()])
    with pytest.raises(ValueError, match="image_fraction"):
        augment_samples_with_images(
            samples,
            image_fraction=1.5,
            image_count=1,
            image_long_side=512,
            image_aspect_ratio=1.0,
        )
