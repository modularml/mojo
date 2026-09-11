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

"""End-to-end test for MXTOOLS-398 image mixing.

Ties together the pieces added across the stack: SessionMessage carrying
images, dataset-agnostic augmentation, the multi-turn driver attaching
images to outgoing requests, and --dry-run workload stats reporting them.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from max.benchmark.benchmark_shared.config import SamplingConfig
from max.benchmark.benchmark_shared.datasets._tokenizer_pool import (
    TokenizerPool,
)
from max.benchmark.benchmark_shared.datasets.image_augmentation import (
    augment_samples_with_images,
)
from max.benchmark.benchmark_shared.datasets.multiturn_distribution_fit import (
    build_fitted_chat_samples,
)
from max.benchmark.benchmark_shared.datasets.types import (
    ChatMessage,
    ImageContentBlock,
    OpenAIImage,
    TextContentBlock,
)
from max.benchmark.benchmark_shared.multi_turn import chat_session_driver
from max.benchmark.benchmark_shared.request import (
    BaseRequestFuncInput,
    OpenAIChatCompletionsRequestDriver,
    RequestCounter,
    RequestDriver,
    RequestFuncInput,
    RequestFuncOutput,
)
from max.benchmark.benchmark_shared.serving_result_output import (
    print_workload_stats,
)
from pytest_mock import MockerFixture


class _FakeTokenizer:
    """Picklable stand-in for `PreTrainedTokenizerBase`, mirroring the one in
    test_benchmark_datasets.py."""

    name_or_path = "_fake_"
    vocab_size = 1000
    unk_token_id = None
    all_special_ids: frozenset[int] = frozenset({0, 1, 2})

    def __init__(self, model_max_length: int = 50_000) -> None:
        self.model_max_length = model_max_length

    def encode(
        self, text: str, add_special_tokens: bool = False, **_: object
    ) -> list[int]:
        return list(range(max(4, len(text))))

    def decode(
        self, ids: list[int], skip_special_tokens: bool = False, **_: object
    ) -> str:
        return "Z" * len(ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return 223


def _fake_loader(
    name_or_path: str,
    model_max_length: int | None,
    trust_remote_code: bool,
    revision: str | None,
) -> _FakeTokenizer:
    return _FakeTokenizer(model_max_length=model_max_length or 50_000)


class _CapturingDriver(RequestDriver):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[RequestFuncInput] = []

    async def request(
        self, request_func_input: BaseRequestFuncInput
    ) -> RequestFuncOutput:
        assert isinstance(request_func_input, RequestFuncInput)
        self.calls.append(request_func_input)
        return RequestFuncOutput(
            success=True,
            latency=0.1,
            ttft=0.05,
            prompt_len=request_func_input.prompt_len,
            generated_text="ok",
        )


def test_image_mixing_pipeline_end_to_end(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A dataset-agnostic multi-turn workload augmented with images: the
    images survive sampling, get attached to the wire request by the
    multi-turn driver, and show up in --dry-run workload stats."""
    tok = _FakeTokenizer()
    pool_texts = [f"turn body {i} " * 20 for i in range(6)]

    with TokenizerPool(tok, loader=_fake_loader) as pool:
        samples = build_fitted_chat_samples(
            pool=pool,
            user_text_pool=pool_texts,
            num_sessions=3,
            num_turns="2",
            input_len="80",
            output_len="20",
            delay_between_turns_dist=None,
            sys_prompt_ratio=0.0,
            max_num_unique_sys_prompt=1,
            shuffle_pool=False,
            log_prefix="test-integration",
        )

    augment_samples_with_images(
        samples,
        image_fraction=1.0,
        image_count=1,
        image_long_side=64,
        image_aspect_ratio=1.0,
        image_turn="first",
    )

    # The first user turn of every session should carry exactly one image.
    for session in samples.chat_sessions:
        user_messages = [m for m in session.messages if m.source == "user"]
        assert len(user_messages[0].images) == 1
        for later in user_messages[1:]:
            assert len(later.images) == 0

    driver = _CapturingDriver()

    async def run_first_session() -> None:
        await chat_session_driver(
            model_id="test-model",
            api_url="http://localhost:8000/v1/chat/completions",
            request_driver=driver,
            request_counter=RequestCounter(max_requests=10),
            chat_session=samples.chat_sessions[0],
            max_chat_len=100_000,
            sampling=SamplingConfig(),
        )

    asyncio.run(run_first_session())

    assert len(driver.calls) >= 1
    first_call = driver.calls[0]
    assert isinstance(first_call.prompt, list)
    first_user_content = first_call.prompt[0].content
    assert any(
        isinstance(block, ImageContentBlock) for block in first_user_content
    )

    print_workload_stats(samples)
    out = capsys.readouterr().out
    assert "Image count (per session)" in out


@pytest.mark.asyncio
async def test_serialized_payload_carries_the_image(
    mocker: MockerFixture,
) -> None:
    """Assert the wire payload, not just the in-memory prompt.

    Checking the `ChatMessage` objects alone cannot catch an endpoint that
    drops images during serialization, which is the failure this whole
    feature is exposed to.
    """
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    image: OpenAIImage = {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,AAA"},
    }
    request_input = RequestFuncInput(
        model="test-model",
        session_id=None,
        sampling=SamplingConfig(),
        prompt=[
            ChatMessage(role="system", content=[TextContentBlock(text="sys")]),
            ChatMessage(role="user", content=[TextContentBlock(text="hi")]),
        ],
        images=[image],
        api_url="http://localhost:8000/v1/chat/completions",
        prompt_len=10,
        max_tokens=16,
        ignore_eos=False,
    )

    session_class = mocker.patch(
        "max.benchmark.benchmark_shared.request.aiohttp.ClientSession"
    )
    client = session_class.return_value.__aenter__.return_value

    async def body() -> AsyncIterator[bytes]:
        yield b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    response = mocker.AsyncMock()
    response.status = 200
    response.content = body()
    post_ctx = mocker.AsyncMock()
    post_ctx.__aenter__ = mocker.AsyncMock(return_value=response)
    post_ctx.__aexit__ = mocker.AsyncMock(return_value=None)
    client.post = mocker.Mock(return_value=post_ctx)

    await OpenAIChatCompletionsRequestDriver().request(request_input)

    payload = client.post.call_args.kwargs["json"]
    roles = [m["role"] for m in payload["messages"]]
    user_message = payload["messages"][roles.index("user")]
    assert image in user_message["content"], (
        "the serialized payload must carry the image"
    )
    system_message = payload["messages"][roles.index("system")]
    assert image not in system_message["content"], (
        "the image must not land on the system message"
    )
