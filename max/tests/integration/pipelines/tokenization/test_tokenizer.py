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

import asyncio
import json
import pickle
from unittest.mock import MagicMock, NonCallableMock

import numpy as np
import pytest
import requests
from max.driver import DeviceSpec, accelerator_count
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineConfig,
    TextAndVisionTokenizer,
    TextTokenizer,
)
from max.pipelines.context import (
    SamplingParams,
    TextAndVisionContext,
    TextContext,
    TextGenerationResponseFormat,
    TokenBuffer,
    validate_only_one_image,
)
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, SamplingConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.lib.tokenizer import replace_unpaired_surrogates
from max.pipelines.modeling.types import (
    ImageContentPart,
    MessageContent,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestFunction,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from test_common.mocks import mock_plan_from_sizes
from transformers import AutoConfig


def _create_mock_pipeline_config(model_path: str) -> MagicMock:
    """Create a mock PipelineConfig with real HuggingFace config."""
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    mock_kv_cache_config = NonCallableMock(spec=KVCacheConfig)
    mock_kv_cache_config.enable_prefix_caching = False

    mock_model_config = MagicMock()
    mock_model_config.huggingface_config = hf_config
    mock_model_config.kv_cache = mock_kv_cache_config

    pipeline_config = MagicMock()
    pipeline_config.model = mock_model_config
    return pipeline_config


LLAMA_3_1_HF_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"


def convert_image_url_to_base64(image_url: str) -> bytes | None:
    """Fetches an image from a URL and converts it to Base64 encoded bytes."""
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error for bad responses
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image: {e}")
        return None


@pytest.mark.skip("this requires authorized huggingface access")
def test_text_and_vision_tokenizer() -> None:
    """This test uses gated repos on huggingface, as such its not expected to run in CI.
    It is primarily written to test out the chat templating for multi-modal models.
    """

    VALID_REPOS = {
        # This is not currently working for pixtral.
        "mistral-experimental/pixtral-12b": "[IMG]",
    }
    img_url = "https://picsum.photos/id/237/200/300"
    img = convert_image_url_to_base64(img_url)
    imgs = [[], [img], [img, img]]
    for repo_id in VALID_REPOS:
        model_path = repo_id
        pipeline_config = _create_mock_pipeline_config(model_path)
        tokenizer = TextAndVisionTokenizer(
            model_path, pipeline_config=pipeline_config, trust_remote_code=True
        )
        for imgs_list in imgs:
            content: list[MessageContent] = []
            content.append(TextContentPart(text="What is in this image?"))
            content.extend([ImageContentPart() for _ in imgs_list])
            filtered_imgs_list = [img for img in imgs_list if img is not None]
            assert len(filtered_imgs_list) == len(imgs_list)
            request = TextGenerationRequest(
                request_id=RequestID("request..."),
                model_name=repo_id,
                messages=[
                    TextGenerationRequestMessage(
                        role="user",
                        content=content,
                    )
                ],
                images=filtered_imgs_list,
            )

            context: TextAndVisionContext = asyncio.run(
                tokenizer.new_context(request)
            )

            if not imgs_list:
                assert len(context.images) == 0
            else:
                assert len(context.images) == len(imgs_list)


@pytest.mark.skip("CI does not appear to be working well with gated repos")
def test_text_tokenizer_with_tool_use(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    """This test uses gated repos on huggingface, as such its not expected to run in CI.
    It is written to test out chat templating and input features for tool use with Llama 3.2
    """

    model_path = llama_3_1_8b_instruct_local_path
    pipeline_config = _create_mock_pipeline_config(model_path)
    tokenizer = TextTokenizer(model_path, pipeline_config=pipeline_config)

    request = TextGenerationRequest(
        request_id=RequestID("request_with_tools"),
        model_name=model_path,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content="What is the weather in Toronto?",
            )
        ],
        tools=[
            TextGenerationRequestTool(
                type="function",
                function=TextGenerationRequestFunction(
                    name="get_current_weather",
                    description="Get the current weather for a given location.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. Toronto.",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                ),
            )
        ],
    )

    asyncio.run(tokenizer.new_context(request))


def test_tokenizer__truncates_to_max_length(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    max_length = 12
    pipeline_config = _create_mock_pipeline_config(
        llama_3_1_8b_instruct_local_path
    )
    tokenizer = TextTokenizer(
        llama_3_1_8b_instruct_local_path,
        pipeline_config=pipeline_config,
        max_length=max_length,
    )

    short_request = TextGenerationRequest(
        request_id=RequestID("request_with_short_message"),
        model_name=llama_3_1_8b_instruct_local_path,
        prompt="Short message",
    )
    context: TextContext = asyncio.run(tokenizer.new_context(short_request))
    assert len(context.tokens) < 12

    long_request = TextGenerationRequest(
        request_id=RequestID("request_with_short_message"),
        model_name=llama_3_1_8b_instruct_local_path,
        prompt="Longer message with lots of text with more words than max length for sure.",
    )
    with pytest.raises(ValueError, match="maximum context length"):
        _ = asyncio.run(tokenizer.new_context(long_request))


def test_tokenizer__with_prompt_as_list_of_int(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    pipeline_config = _create_mock_pipeline_config(
        llama_3_1_8b_instruct_local_path
    )
    tokenizer = TextTokenizer(
        llama_3_1_8b_instruct_local_path, pipeline_config=pipeline_config
    )
    request = TextGenerationRequest(
        request_id=RequestID(),
        model_name=llama_3_1_8b_instruct_local_path,
        prompt=[0, 1, 2, 3, 4, 5],
    )
    context = asyncio.run(tokenizer.new_context(request))
    assert np.array_equal(context.tokens.all, np.array([0, 1, 2, 3, 4, 5]))


# Built via ``json.loads`` to mirror the real request path: a client may send
# an emoji whose UTF-16 surrogate pair was split by truncation, which the JSON
# parser accepts as a lone surrogate. ``\ude00`` is a lone low surrogate,
# ``\ud800`` a lone high surrogate; ``\ud83d\ude00`` is the escaped
# surrogate pair the JSON parser reconstitutes into the single
# grinning-face emoji (U+1F600) -- the same decode path a real request
# takes. See CENG-790.
_LONE_LOW = json.loads(r'"\ude00"')
_LONE_HIGH = json.loads(r'"\ud800"')
_CONSECUTIVE_LONE = json.loads(r'"\ud800\ud800"')
_MIXED = json.loads(r'"hi \ude00 there"')
_VALID_PAIR = json.loads(r'"\ud83d\ude00"')


def test_replace_unpaired_surrogates() -> None:
    # Each unpaired surrogate collapses to U+FFFD; surrounding text survives.
    assert replace_unpaired_surrogates(_LONE_LOW) == "\ufffd"
    assert replace_unpaired_surrogates(_LONE_HIGH) == "\ufffd"
    assert replace_unpaired_surrogates(_CONSECUTIVE_LONE) == "\ufffd\ufffd"
    assert replace_unpaired_surrogates(_MIXED) == "hi \ufffd there"

    # Well-formed text is returned unchanged, including a reconstituted pair.
    for well_formed in ("", "hello", "héllo 世界", _VALID_PAIR):
        assert replace_unpaired_surrogates(well_formed) == well_formed

    # The result is always encodable as UTF-8 (the property the tokenizer needs).
    for text in (_LONE_LOW, _LONE_HIGH, _CONSECUTIVE_LONE, _MIXED):
        replace_unpaired_surrogates(text).encode("utf-8")


@pytest.mark.asyncio
async def test_tokenizer_encode_handles_lone_surrogates(
    smollm_135m_local_path: str,
) -> None:
    pipeline_config = _create_mock_pipeline_config(smollm_135m_local_path)
    tokenizer = TextTokenizer(
        smollm_135m_local_path, pipeline_config=pipeline_config
    )

    # Guard the premise: the underlying fast tokenizer rejects a lone surrogate
    # with the opaque TypeError this fix exists to prevent.
    assert tokenizer.delegate.is_fast
    with pytest.raises(TypeError):
        tokenizer.delegate.encode(_LONE_LOW)

    # The wrapper normalizes first, so encoding succeeds for every variant.
    for text in (_LONE_LOW, _LONE_HIGH, _CONSECUTIVE_LONE, _MIXED):
        encoded = await tokenizer.encode(text, add_special_tokens=False)
        assert len(encoded) > 0

    # Well-formed input is unaffected: a valid pair encodes identically to the
    # emoji string it represents.
    assert np.array_equal(
        await tokenizer.encode(_VALID_PAIR, add_special_tokens=False),
        await tokenizer.encode("\U0001f600", add_special_tokens=False),
    )


@pytest.mark.asyncio
async def test_tokenizer_new_context_handles_lone_surrogate(
    smollm_135m_local_path: str,
) -> None:
    pipeline_config = _create_mock_pipeline_config(smollm_135m_local_path)
    tokenizer = TextTokenizer(
        smollm_135m_local_path, pipeline_config=pipeline_config
    )
    request = TextGenerationRequest(
        request_id=RequestID(),
        model_name=smollm_135m_local_path,
        prompt=_MIXED,
    )
    context = await tokenizer.new_context(request)
    assert len(context.tokens) > 0


def test_tokenizer__propagates_cache_salt(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    """`tokenizer.new_context` must forward `request.cache_salt` to
    `context.cache_salt` (Step 6.5 plumbing). Without this the
    multi-tenant prefix-cache isolation feature is silently inert."""

    pipeline_config = _create_mock_pipeline_config(
        llama_3_1_8b_instruct_local_path
    )
    tokenizer = TextTokenizer(
        llama_3_1_8b_instruct_local_path,
        pipeline_config=pipeline_config,
    )

    request_with_salt = TextGenerationRequest(
        request_id=RequestID(),
        model_name=llama_3_1_8b_instruct_local_path,
        prompt="Hello world!",
        cache_salt="tenant-abc",
    )

    context_with_salt = asyncio.run(tokenizer.new_context(request_with_salt))
    assert context_with_salt.cache_salt == "tenant-abc"

    request_without_salt = TextGenerationRequest(
        request_id=RequestID(),
        model_name=llama_3_1_8b_instruct_local_path,
        prompt="Hello world!",
    )

    context_without_salt = asyncio.run(
        tokenizer.new_context(request_without_salt)
    )
    assert context_without_salt.cache_salt is None


def test_tokenizer__propagates_dkv_cache_hint(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    """`tokenizer.new_context` must re-serialize `request.dkv_cache_hint` onto
    `context.dkv_cache_hint`. The dKV connector parses those bytes to route a
    load at its peers, so dropping them here silently disables remote reads."""

    pipeline_config = _create_mock_pipeline_config(
        llama_3_1_8b_instruct_local_path
    )
    tokenizer = TextTokenizer(
        llama_3_1_8b_instruct_local_path,
        pipeline_config=pipeline_config,
    )
    hint = {"version": 2, "instances": [{"instance_name": "dkv-peer"}]}

    context = asyncio.run(
        tokenizer.new_context(
            TextGenerationRequest(
                request_id=RequestID(),
                model_name=llama_3_1_8b_instruct_local_path,
                prompt="Hello world!",
                dkv_cache_hint=hint,
            )
        )
    )
    assert context.dkv_cache_hint is not None
    assert json.loads(context.dkv_cache_hint) == hint

    context_without_hint = asyncio.run(
        tokenizer.new_context(
            TextGenerationRequest(
                request_id=RequestID(),
                model_name=llama_3_1_8b_instruct_local_path,
                prompt="Hello world!",
            )
        )
    )
    assert context_without_hint.dkv_cache_hint is None


def test_tokenizer__with_context_validation(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    from max.pipelines.lib.registry import _apply_context_validators

    def raise_fn(context: TextContext | TextAndVisionContext) -> None:
        raise ValueError("test")

    pipeline_config = _create_mock_pipeline_config(
        llama_3_1_8b_instruct_local_path
    )
    tokenizer = TextTokenizer(
        llama_3_1_8b_instruct_local_path,
        pipeline_config=pipeline_config,
        trust_remote_code=True,
    )
    _apply_context_validators(tokenizer, [raise_fn])

    request = TextGenerationRequest(
        request_id=RequestID("request_with_short_message"),
        model_name=llama_3_1_8b_instruct_local_path,
        prompt="Short message",
    )

    with pytest.raises(ValueError, match="test"):
        _ = asyncio.run(tokenizer.new_context(request))


def test_tokenizer__with_context_validators_is_pickleable(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    """Tokenizers must be pickleable because MAX Serve pickles them when
    spawning model workers (multiprocessing with 'spawn' start method).
    """
    pipeline_config = _create_mock_pipeline_config(
        llama_3_1_8b_instruct_local_path
    )
    tokenizer = TextTokenizer(
        llama_3_1_8b_instruct_local_path,
        pipeline_config=pipeline_config,
        trust_remote_code=True,
        context_validators=[validate_only_one_image],
    )

    # Round-trip through pickle: this is what multiprocessing 'spawn' does.
    restored = pickle.loads(pickle.dumps(tokenizer))

    # Verify the restored tokenizer still works.
    request = TextGenerationRequest(
        request_id=RequestID("pickle_roundtrip"),
        model_name=llama_3_1_8b_instruct_local_path,
        prompt="Short message",
    )
    context = asyncio.run(restored.new_context(request))
    assert len(context.tokens) > 0


def test_tokenizer_regression_MODELS_467() -> None:
    """Regression test for
    https://linear.app/modularml/issue/MODELS-467/[bug]-no-text-response-mistralaimistral-7b-instruct-v03
    """
    model_path = "mistralai/Mistral-7B-Instruct-v0.3"
    pipeline_config = _create_mock_pipeline_config(model_path)
    tokenizer = TextTokenizer(
        model_path,
        pipeline_config=pipeline_config,
        revision="e0bc86c23ce5aae1db576c8cca6f06f1f73af2db",
        enable_llama_whitespace_fix=True,
        trust_remote_code=True,
    )

    def rank1(items: list[int]) -> np.ndarray:
        return np.array(items, dtype=np.uint32)

    def rank0(item: int) -> np.ndarray:
        return rank1([item])[0]

    def decode(tokens: np.ndarray) -> str:
        return asyncio.run(tokenizer.decode(tokens))

    # Single token here needs preceding space, including rank-0.
    assert decode(rank1([23325])) == " Hello"
    assert decode(rank0(23325)) == " Hello"
    assert decode(rank1([23325, 2294])) == "Hello world"
    # But not all single tokens should have a preceding space, including rank-0.
    assert decode(rank1([1056])) == "ing"
    assert decode(rank0(1056)) == "ing"


@pytest.mark.asyncio
async def test_tokenizer__encode_and_decode(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    pipeline_config = _create_mock_pipeline_config(
        llama_3_1_8b_instruct_local_path
    )
    tokenizer = TextTokenizer(
        model_path=llama_3_1_8b_instruct_local_path,
        pipeline_config=pipeline_config,
    )

    test_string = "hi my name is"
    encoded = await tokenizer.encode(test_string, add_special_tokens=False)
    context = TextContext(
        max_length=10,
        tokens=TokenBuffer(np.array(encoded, dtype=np.int64)),
    )
    assert len(context.tokens) == len(encoded)
    decoded = await tokenizer.decode(encoded)
    assert test_string == decoded


@pytest.mark.skip("TODO: Fix this flaky test")
@mock_plan_from_sizes
def test_text_tokenizer_with_constrained_decoding(
    modular_ai_llama_3_1_local_path: str,
) -> None:
    device_specs = []
    if accelerator_count() > 0:
        device_specs.append(DeviceSpec.accelerator(id=0))
    else:
        device_specs.append(DeviceSpec.cpu(id=0))
    pipeline_config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=modular_ai_llama_3_1_local_path,
                    quantization_encoding="bfloat16",
                    device_specs=device_specs,
                )
            }
        ),
        sampling=SamplingConfig(enable_structured_output=True),
    )

    tokenizer = TextTokenizer(
        pipeline_config.model.model_path, pipeline_config=pipeline_config
    )

    prompt = """
    Please provide a json response, with the person's name and age extracted from the excerpt.
    For example, provided an excerpt 'Joe Biden is 100 years old.' return with {"name": "Joe Biden", "age": 100}.

    Please extract the person's name and age from the following excerpt:
    'Donald Trump is 102 years old.'

    """

    request = TextGenerationRequest(
        request_id=RequestID("request_with_tools"),
        model_name=pipeline_config.model.model_path,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=prompt,
            )
        ],
        response_format=TextGenerationResponseFormat(
            type="json_schema",
            grammar=None,
            json_schema={
                "title": "Person",
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                    },
                    "age": {
                        "type": "integer",
                    },
                },
                "required": ["name", "age"],
            },
            grammar_enforced=True,
            tools_forced=False,
        ),
    )

    context = asyncio.run(tokenizer.new_context(request))

    assert context.json_schema


def test_tokenizer_encode_stop_criteria(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    pipeline_config = _create_mock_pipeline_config(
        llama_3_1_8b_instruct_local_path
    )
    tokenizer = TextTokenizer(
        model_path=llama_3_1_8b_instruct_local_path,
        pipeline_config=pipeline_config,
    )

    prompt = "hi my name is"

    request = TextGenerationRequest(
        request_id=RequestID("id_0"),
        model_name=llama_3_1_8b_instruct_local_path,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content=prompt,
            )
        ],
        sampling_params=SamplingParams(stop=["!"]),
    )

    context = asyncio.run(tokenizer.new_context(request))
    # encoded stop criteria should equal [0]
    assert len(context.eos_tracker.eos_sequences) == 1
    assert len(context.eos_tracker.eos_sequences[0]) == 1
    assert np.array_equal(context.eos_tracker.eos_sequences[0], [0])


@pytest.mark.skip("TODO: test fails on 4xH100 CI")
def test_text_and_vision_tokenizer_forwards_sampling_params() -> None:
    """Test that TextAndVisionTokenizer properly forwards sampling params to context."""
    model_path = "OpenGVLab/InternVL3-1B-Instruct"

    pipeline_config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=model_path,
                    trust_remote_code=True,
                )
            }
        ),
    )

    tokenizer = PIPELINE_REGISTRY.retrieve_tokenizer(pipeline_config)

    custom_params = SamplingParams(
        temperature=0.5,
        top_k=42,
    )

    request = TextGenerationRequest(
        request_id=RequestID("test_request"),
        model_name=model_path,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content="test message",
            )
        ],
        sampling_params=custom_params,
    )

    context = asyncio.run(tokenizer.new_context(request))

    assert context.sampling_params is not None
    assert context.sampling_params.temperature == 0.5
    assert context.sampling_params.top_k == 42


def test_tokenizer_stores_eos_token_ids(
    modular_ai_llama_3_1_local_path: str,
) -> None:
    """Tests that all eos token ids stored in the huggingface config are added
    to the tokenizer's eos token ids.
    """
    # Must pass in PipelineConfig so the tokenizer can access the
    # huggingface config.
    device_specs = []
    if accelerator_count() > 0:
        device_specs.append(DeviceSpec.accelerator(id=0))
    else:
        device_specs.append(DeviceSpec.cpu(id=0))
    pipeline_config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=modular_ai_llama_3_1_local_path,
                    quantization_encoding="bfloat16",
                    device_specs=device_specs,
                )
            }
        ),
    )

    # Test single eos token id
    assert pipeline_config.model.huggingface_config is not None
    pipeline_config.model.huggingface_config.eos_token_id = 123456
    tokenizer = TextTokenizer(
        model_path=modular_ai_llama_3_1_local_path,
        pipeline_config=pipeline_config,
    )
    assert tokenizer.eos_token_ids == {tokenizer.delegate.eos_token_id, 123456}

    # Test list of eos token ids
    assert pipeline_config.model.huggingface_config is not None
    pipeline_config.model.huggingface_config.eos_token_id = [123, 456]
    tokenizer = TextTokenizer(
        model_path=modular_ai_llama_3_1_local_path,
        pipeline_config=pipeline_config,
    )
    assert tokenizer.eos_token_ids == {
        tokenizer.delegate.eos_token_id,
        123,
        456,
    }


def test_text_and_vision_tokenizer_stores_eos_token_ids(
    google_gemma_3_4b_it_local_path: str,
) -> None:
    """Tests that all eos token ids stored in the huggingface config are added
    to the TextAndVisionTokenizer's eos token ids.
    """
    model_path = google_gemma_3_4b_it_local_path

    pipeline_config = PipelineConfig(
        models=ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path=model_path,
                    trust_remote_code=True,
                    max_length=100,
                )
            }
        ),
        runtime=PipelineRuntimeConfig(max_batch_size=1),
    )

    gemma_3_eos_token_ids = {1, 106}
    tokenizer = TextAndVisionTokenizer(
        model_path=model_path,
        pipeline_config=pipeline_config,
    )
    assert tokenizer.eos_token_ids == gemma_3_eos_token_ids


@pytest.mark.asyncio
async def test_tokenizer__apply_chat_template_dict_list_vs_str_content(
    llama_3_1_8b_instruct_local_path,  # noqa: ANN001
) -> None:
    pipeline_config = _create_mock_pipeline_config(
        llama_3_1_8b_instruct_local_path
    )
    tokenizer = TextTokenizer(
        model_path=llama_3_1_8b_instruct_local_path,
        pipeline_config=pipeline_config,
    )

    messages = [
        TextGenerationRequestMessage(
            role="user",
            content=[
                TextContentPart(text="Hello, how are you"),
                TextContentPart(text="today?"),
            ],
        ),
        TextGenerationRequestMessage(
            role="assistant",
            content="I'm doing well, thank you!",
        ),
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tools=None)
    expected_prompt_text = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\n"
        "Today Date: 26 Jul 2024\n\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "Hello, how are you\ntoday?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "I'm doing well, thank you!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert prompt_text == expected_prompt_text


@pytest.mark.asyncio
async def test_tokenizer__apply_chat_template_preserves_default_options(
    llama_3_1_8b_instruct_local_path,  # noqa: ANN001
) -> None:
    """Caller-provided chat_template_options must not drop the implicit
    add_generation_prompt=True default. Regression test: OpenRouter-style
    callers send `chat_template_kwargs: {"thinking": true}` and expect the
    assistant turn prefix to still be appended."""
    pipeline_config = _create_mock_pipeline_config(
        llama_3_1_8b_instruct_local_path
    )
    tokenizer = TextTokenizer(
        model_path=llama_3_1_8b_instruct_local_path,
        pipeline_config=pipeline_config,
    )
    messages = [
        TextGenerationRequestMessage(role="user", content="Hi"),
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tools=None,
        chat_template_options={"date_string": "26 Jul 2024"},
    )

    assert prompt_text.endswith(
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


@pytest.mark.asyncio
async def test_tokenizer__generate_prompt_and_token_ids(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    """Test the _generate_prompt_and_token_ids method of TextTokenizer."""
    pipeline_config = _create_mock_pipeline_config(
        llama_3_1_8b_instruct_local_path
    )
    tokenizer = TextTokenizer(
        model_path=llama_3_1_8b_instruct_local_path,
        pipeline_config=pipeline_config,
    )

    # Test with string prompt
    prompt = "Hello, how are you?"
    prompt_text, token_ids = await tokenizer._generate_prompt_and_token_ids(
        prompt=prompt,
        messages=[],
    )
    assert isinstance(prompt_text, str)
    assert prompt_text == prompt
    assert isinstance(token_ids, np.ndarray)
    expected_token_ids = await tokenizer.encode(prompt, add_special_tokens=True)
    assert np.array_equal(token_ids, expected_token_ids)

    # Test with list of messages
    messages = [
        TextGenerationRequestMessage(
            role="user",
            content="Hello, how are you?",
        ),
        TextGenerationRequestMessage(
            role="assistant",
            content="I'm doing well, thank you!",
        ),
    ]
    prompt_text, token_ids = await tokenizer._generate_prompt_and_token_ids(
        prompt=None,
        messages=messages,
    )
    assert isinstance(prompt_text, str)
    expected_token_ids = await tokenizer.encode(
        prompt_text, add_special_tokens=False
    )
    assert np.array_equal(token_ids, expected_token_ids)
    # Verify that the chat template was applied
    assert "Hello, how are you?" in prompt_text
    assert "I'm doing well, thank you!" in prompt_text


@pytest.mark.asyncio
async def test_custom_prompt_template() -> None:
    """Test that custom prompt_template parameter overrides the model's default template."""
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

    # Define a simple custom template for testing
    custom_template = """{% for message in messages %}
{% if message['role'] == 'user' %}
USER: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
ASSISTANT: {{ message['content'] }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
ASSISTANT: {% endif %}"""

    # Create tokenizer with custom template
    pipeline_config = _create_mock_pipeline_config(model_name)
    tokenizer = TextTokenizer(
        model_path=model_name,
        pipeline_config=pipeline_config,
        trust_remote_code=True,
        chat_template=custom_template,
    )

    # Test messages
    messages = [
        TextGenerationRequestMessage(
            role="user",
            content="Hello, how are you?",
        )
    ]

    # Generate prompt with custom template
    generated_prompt = tokenizer.apply_chat_template(
        messages=messages,
        tools=None,
        chat_template_options={"add_generation_prompt": True},
    )

    # Verify the custom template was used
    assert "USER: Hello, how are you?" in generated_prompt, (
        f"Custom template not used. Generated prompt: {generated_prompt}"
    )
    assert "ASSISTANT:" in generated_prompt, (
        f"Custom template not used. Generated prompt: {generated_prompt}"
    )

    # Compare with default template to ensure it's different
    default_tokenizer = TextTokenizer(
        model_path=model_name,
        pipeline_config=pipeline_config,
        trust_remote_code=True,
    )

    default_prompt = default_tokenizer.apply_chat_template(
        messages=messages,
        tools=None,
        chat_template_options={"add_generation_prompt": True},
    )

    # Custom and default prompts should be different
    assert generated_prompt != default_prompt, (
        f"Custom template produced same result as default.\n"
        f"Custom: {generated_prompt}\n"
        f"Default: {default_prompt}"
    )


@pytest.mark.asyncio
async def test_custom_prompt_template_beats_tokenizer_override() -> None:
    """A tokenizer that renders its own format must not defeat --chat-template.

    ``trust_remote_code`` tokenizers (e.g. Kimi's ``TikTokenTokenizer``)
    override ``apply_chat_template`` with a format hardcoded in Python and
    never read the ``chat_template`` attribute, which silently discarded the
    override.
    """
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    custom_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )

    pipeline_config = _create_mock_pipeline_config(model_name)
    tokenizer = TextTokenizer(
        model_path=model_name,
        pipeline_config=pipeline_config,
        trust_remote_code=True,
        chat_template=custom_template,
    )
    tokenizer.delegate.apply_chat_template = lambda *args, **kwargs: (
        "HARDCODED FORMAT"
    )

    prompt = tokenizer.apply_chat_template(
        messages=[
            TextGenerationRequestMessage(role="user", content="Hello there")
        ],
        tools=None,
    )

    assert prompt == "Hello there"


@pytest.mark.asyncio
async def test_custom_prompt_template_with_tools() -> None:
    """Test that custom prompt_template works with tools."""
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

    # Define a custom template that includes tools
    custom_template = """{% for message in messages %}
{% if message['role'] == 'user' %}
USER: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
ASSISTANT: {{ message['content'] }}
{% endif %}
{% endfor %}
{% if tools %}
TOOLS AVAILABLE:
{% for tool in tools %}
- {{ tool.function.name }}: {{ tool.function.description }}
{% endfor %}
{% endif %}
{% if add_generation_prompt %}
ASSISTANT: {% endif %}"""

    # Create tokenizer with custom template
    pipeline_config = _create_mock_pipeline_config(model_name)
    tokenizer = TextTokenizer(
        model_path=model_name,
        pipeline_config=pipeline_config,
        trust_remote_code=True,
        chat_template=custom_template,
    )

    # Test with tools
    tools = [
        TextGenerationRequestTool(
            type="function",
            function=TextGenerationRequestFunction(
                name="get_weather",
                description="Get the current weather for a location.",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get weather for",
                        }
                    },
                    "required": ["location"],
                },
            ),
        )
    ]

    messages = [
        TextGenerationRequestMessage(
            role="user",
            content="What's the weather like?",
        )
    ]

    # Generate prompt with custom template and tools
    generated_prompt = tokenizer.apply_chat_template(
        messages=messages,
        tools=tools,
        chat_template_options={"add_generation_prompt": True},
    )

    # Verify the custom template was used with tools
    assert "TOOLS AVAILABLE:" in generated_prompt, (
        f"Custom template with tools not used. Generated prompt: {generated_prompt}"
    )
    assert "get_weather:" in generated_prompt, (
        f"Tool not included in custom template. Generated prompt: {generated_prompt}"
    )
    assert "USER: What's the weather like?" in generated_prompt, (
        f"User message not properly formatted. Generated prompt: {generated_prompt}"
    )


@pytest.mark.asyncio
async def test_custom_prompt_template_error_handling() -> None:
    """Test error handling for broken custom prompt templates."""
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

    # Create a deliberately broken template (missing closing tag)
    broken_template = """{% for message in messages %}
{% if message['role'] == 'user' %}
USER: {{ message['content']
{% elif message['role'] == 'assistant' %}
ASSISTANT: {{ message['content'] }}
{% endif %}
{% endfor %}"""  # Note: missing closing }} for first message['content']

    # Create tokenizer with broken template
    pipeline_config = _create_mock_pipeline_config(model_name)
    tokenizer = TextTokenizer(
        model_path=model_name,
        pipeline_config=pipeline_config,
        trust_remote_code=True,
        chat_template=broken_template,
    )

    messages = [
        TextGenerationRequestMessage(
            role="user",
            content="Hello",
        )
    ]

    # This should raise a ValueError with helpful context
    with pytest.raises(ValueError) as exc_info:
        tokenizer.apply_chat_template(
            messages=messages,
            tools=None,
            chat_template_options={"add_generation_prompt": True},
        )

    error_message = str(exc_info.value)

    # Verify the error message contains the expected guidance
    assert "Failed to apply custom chat template" in error_message
    assert "Template variables available" in error_message
    assert "messages: List of conversation messages" in error_message
    assert "Original error" in error_message


@pytest.mark.asyncio
async def test_default_template_error_passthrough() -> None:
    """Test that default template errors are passed through unchanged."""
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

    # Create tokenizer without custom template
    pipeline_config = _create_mock_pipeline_config(model_name)
    tokenizer = TextTokenizer(
        model_path=model_name,
        pipeline_config=pipeline_config,
        trust_remote_code=True,
    )

    # Create an edge case that might cause an error with some models
    # This test verifies that we don't wrap default template errors
    empty_messages: list[TextGenerationRequestMessage] = []

    try:
        # Try to apply chat template with empty messages
        tokenizer.apply_chat_template(
            messages=empty_messages,
            tools=None,
            chat_template_options={"add_generation_prompt": True},
        )
        # If it doesn't error, that's also fine - just means the model handles empty messages
    except Exception as e:
        # If it does error, ensure it's NOT wrapped with our custom error message
        error_str = str(e)
        assert "Failed to apply custom chat template" not in error_str, (
            f"Default template error was incorrectly wrapped: {error_str}"
        )


@pytest.mark.asyncio
async def test_custom_template_new_context_integration() -> None:
    """Test that custom templates work correctly with new_context method."""
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"

    # Simple custom template
    custom_template = """{% for message in messages %}
{{ message['role']|upper }}: {{ message['content'] }}
{% endfor %}
ASSISTANT: """

    # Create tokenizer with custom template
    pipeline_config = _create_mock_pipeline_config(model_name)
    tokenizer = TextTokenizer(
        model_path=model_name,
        pipeline_config=pipeline_config,
        trust_remote_code=True,
        chat_template=custom_template,
    )

    # Create a request that will use the custom template
    request = TextGenerationRequest(
        request_id=RequestID("test_custom_template"),
        model_name=model_name,
        messages=[
            TextGenerationRequestMessage(
                role="user",
                content="Test message",
            )
        ],
        sampling_params=SamplingParams(
            max_new_tokens=50,
            temperature=0.7,
        ),
    )

    # Create context - this should use our custom template internally
    context = await tokenizer.new_context(request)

    # Verify context was created successfully
    assert context is not None
    assert len(context.tokens) > 0

    # Decode the prompt tokens to verify our custom template was used
    prompt_tokens = context.tokens.prompt
    decoded_prompt = await tokenizer.decode(prompt_tokens)

    # Should contain our custom format
    assert "USER: Test message" in decoded_prompt, (
        f"Custom template not applied in new_context. Decoded prompt: {decoded_prompt}"
    )


@pytest.mark.asyncio
async def test_tokenizer_decode_overflow_error_with_invalid_token_ids(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    """Test that decode raises a helpful OverflowError when token IDs exceed vocab size."""
    pipeline_config = _create_mock_pipeline_config(
        llama_3_1_8b_instruct_local_path
    )
    tokenizer = TextTokenizer(
        model_path=llama_3_1_8b_instruct_local_path,
        pipeline_config=pipeline_config,
    )

    vocab_size = len(tokenizer.delegate.vocab)

    # Create a token array with values exceeding vocab size
    invalid_tokens = np.array(
        [1, 2, vocab_size + 100000, 5, vocab_size + 5000, 281474976710656],
        dtype=np.int64,
    )

    with pytest.raises(OverflowError) as exc_info:
        await tokenizer.decode(invalid_tokens)

    error_message = str(exc_info.value)

    # Verify the error message contains helpful diagnostics
    assert "Invalid token IDs detected" in error_message
    assert "Token IDs exceeding vocab size" in error_message
    assert f"({vocab_size})" in error_message
    assert (
        "[2, 4, 5]" in error_message
    )  # Indices where invalid tokens are located


@pytest.mark.asyncio
async def test_tokenizer_decode_overflow_error_with_negative_token_ids(
    llama_3_1_8b_instruct_local_path: str,
) -> None:
    """Test that decode raises a helpful OverflowError when negative token IDs are present."""
    pipeline_config = _create_mock_pipeline_config(
        llama_3_1_8b_instruct_local_path
    )
    tokenizer = TextTokenizer(
        model_path=llama_3_1_8b_instruct_local_path,
        pipeline_config=pipeline_config,
    )

    # Create a token array with negative values
    negative_tokens = np.array([1, -5, 3, -100, 5], dtype=np.int64)

    with pytest.raises(OverflowError) as exc_info:
        await tokenizer.decode(negative_tokens)

    error_message = str(exc_info.value)

    # Verify the error message contains helpful diagnostics
    assert "Invalid token IDs detected" in error_message
    assert "Negative token IDs at indices" in error_message
    assert (
        "[1, 3]" in error_message
    )  # Indices where negative tokens are located
    assert "[-5, -100]" in error_message  # The actual negative values
