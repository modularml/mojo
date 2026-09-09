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

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from max.graph.weights import WeightsFormat
from max.pipelines import (
    PIPELINE_REGISTRY,
    PipelineArgs,
    PipelineConfig,
    PipelineRuntimeConfig,
)
from max.pipelines.context import TextContext
from max.pipelines.lib.registry import (
    SupportedArchitecture,
    _retrieve_chat_template,
)
from max.pipelines.lib.tokenizer import TextTokenizer
from max.pipelines.modeling.types import PipelineTask
from max.pipelines.weights.hf_utils import (
    generate_local_model_path as _real_generate_local_model_path,
)
from test_common.mocks import (
    mock_pipeline_config_hf_dependencies,
    mock_pipeline_config_resolve,
)
from test_common.pipeline_model_dummy import (
    DUMMY_GEMMA_ARCH,
    DUMMY_LLAMA_ARCH,
    DummyLlamaArchConfig,
    DummyPipelineModel,
    DummyPixelArchConfig,
    DummyPixelTokenizer,
    DummyTextTokenizer,
)
from test_common.registry import prepare_registry


@pytest.fixture(autouse=True)
def _offline_hf_construction() -> Iterator[None]:
    """Keep ``MAXModelConfig`` construction offline (CI runs
    ``HF_HUB_OFFLINE=1``): ``__init__`` eagerly builds the HuggingFace repo
    handles. Real cached repos resolve normally; uncached/placeholder repos
    get a fake path.
    """

    def _gen(repo_id: str, revision: str) -> str:
        try:
            return _real_generate_local_model_path(repo_id, revision)
        except Exception:
            return f"/fake/cache/{repo_id}"

    with (
        patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"),
        patch("max.pipelines.weights.hf_utils.validate_hf_repo_access"),
        patch(
            "max.pipelines.weights.hf_utils.generate_local_model_path",
            side_effect=_gen,
        ),
    ):
        yield


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_registry__test_register() -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
    assert "LlamaForCausalLM" in PIPELINE_REGISTRY.architectures

    # This should fail when registering the architecture for a second time.
    with pytest.raises(ValueError):
        PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_registry__test_retrieve_with_unknown_architecture_max_engine() -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)

    config = PipelineArgs(
        model_path="GSAI-ML/LLaDA-8B-Instruct",
        # This forces it to fail if we don't have it.
        trust_remote_code=True,
        max_length=1,
        runtime=PipelineRuntimeConfig(max_batch_size=1),
    )
    with pytest.raises(ValueError):
        PIPELINE_REGISTRY.retrieve(PipelineConfig.from_args(config))


@prepare_registry
@mock_pipeline_config_hf_dependencies
def test_registry__test_retrieve_with_unknown_architecture_unknown_engine() -> (
    None
):
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)

    config = PipelineArgs(
        model_path="GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        max_length=1,
        runtime=PipelineRuntimeConfig(max_batch_size=1),
    )
    with pytest.raises(
        ValueError,
        match=r"Cannot determine architecture|no 'architectures' field",
    ):
        PIPELINE_REGISTRY.retrieve(PipelineConfig.from_args(config))


@prepare_registry
def test_registry__retrieve_pipeline_task_returns_text_generation() -> None:
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
    task = PIPELINE_REGISTRY.retrieve_pipeline_task("LlamaForCausalLM")
    assert task == PipelineTask.TEXT_GENERATION


@prepare_registry
def test_registry__retrieve_pipeline_task_defaults_to_text_generation_on_ambiguous_architecture() -> (
    None
):
    PIPELINE_REGISTRY.register(DUMMY_LLAMA_ARCH)
    embedding_arch = SupportedArchitecture(
        name="LlamaForCausalLM",
        task=PipelineTask.EMBEDDINGS_GENERATION,
        example_repo_ids=["dummy/embedding-model"],
        default_encoding="bfloat16",
        supported_encodings={"bfloat16"},
        pipeline_model=DummyPipelineModel,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        config=DummyLlamaArchConfig,
    )
    PIPELINE_REGISTRY.register(embedding_arch)
    task = PIPELINE_REGISTRY.retrieve_pipeline_task("LlamaForCausalLM")
    assert task == PipelineTask.TEXT_GENERATION


@prepare_registry
@mock_pipeline_config_resolve
def test_registry__retrieve_factory_pixel_uses_arch_config_max_length() -> None:
    pixel_arch = SupportedArchitecture(
        name="DummyPixelPipeline",
        task=PipelineTask.PIXEL_GENERATION,
        example_repo_ids=["dummy/pixel-model"],
        default_encoding="bfloat16",
        supported_encodings={"bfloat16"},
        pipeline_model=DummyPipelineModel,
        tokenizer=DummyPixelTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        config=DummyPixelArchConfig,
    )
    PIPELINE_REGISTRY.register(pixel_arch)

    pipeline_args = PipelineArgs(
        model_path="dummy/pixel-model",
        quantization_encoding="bfloat16",
        max_length=1,
        runtime=PipelineRuntimeConfig(max_batch_size=1),
    )
    PIPELINE_REGISTRY.retrieve_factory(
        PipelineConfig.from_args(pipeline_args),
        task=PipelineTask.PIXEL_GENERATION,
        override_architecture="DummyPixelPipeline",
    )

    assert DummyPixelTokenizer.init_kwargs["max_length"] == 123


@prepare_registry
@mock_pipeline_config_resolve
def test_registry__retrieve_factory_tokenizer_uses_planned_max_length() -> None:
    """The text tokenizer bound is the memory plan's planned_max_length,
    even when the arch config's get_max_seq_len ignores the user's
    --max-length (the permissive configs return the raw checkpoint bound,
    here 123)."""
    text_arch = SupportedArchitecture(
        name="DummyPermissiveTextModel",
        task=PipelineTask.TEXT_GENERATION,
        example_repo_ids=["dummy/text-model"],
        default_encoding="bfloat16",
        supported_encodings={"bfloat16"},
        pipeline_model=DummyPipelineModel,
        tokenizer=DummyTextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        config=DummyPixelArchConfig,
    )
    PIPELINE_REGISTRY.register(text_arch)

    pipeline_args = PipelineArgs(
        model_path="dummy/text-model",
        quantization_encoding="bfloat16",
        max_length=777,
        runtime=PipelineRuntimeConfig(max_batch_size=1),
    )
    PIPELINE_REGISTRY.retrieve_factory(
        PipelineConfig.from_args(pipeline_args),
        task=PipelineTask.TEXT_GENERATION,
        override_architecture="DummyPermissiveTextModel",
    )

    assert DummyTextTokenizer.init_kwargs["max_length"] == 777


def test_supported_architecture__eq__method() -> None:
    """Test the __eq__ method of SupportedArchitecture class comprehensively."""

    # Create a simple weight adapter function that can be compared
    def simple_adapter(x: Any) -> Any:
        return x

    # Create two identical architectures
    arch1 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding="bfloat16",
        supported_encodings={
            "bfloat16",
            "q4_k",
        },
        pipeline_model=DummyPipelineModel,
        config=DummyLlamaArchConfig,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        weight_adapters={
            WeightsFormat.safetensors: simple_adapter,
            WeightsFormat.gguf: simple_adapter,
        },
        multi_gpu_supported=True,
        required_arguments={"enable_prefix_caching": False},
    )

    arch2 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding="bfloat16",
        supported_encodings={
            "bfloat16",
            "q4_k",
        },
        pipeline_model=DummyPipelineModel,
        config=DummyLlamaArchConfig,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        weight_adapters={
            WeightsFormat.safetensors: simple_adapter,
            WeightsFormat.gguf: simple_adapter,
        },
        multi_gpu_supported=True,
        required_arguments={"enable_prefix_caching": False},
    )

    # Test equality with identical objects
    assert arch1 == arch2
    assert arch2 == arch1

    # Test equality with self
    assert arch1 == arch1  # noqa: PLR0124

    # Test inequality with different class
    assert arch1 != "not an architecture"
    assert arch1 != 42
    assert arch1 is not None

    # Test inequality with different field values
    arch3 = SupportedArchitecture(
        name="DifferentModel",  # Different name
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding="bfloat16",
        supported_encodings={
            "bfloat16",
        },
        pipeline_model=DummyPipelineModel,
        config=DummyLlamaArchConfig,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )
    assert arch1 != arch3

    # Test inequality with different example_repo_ids
    arch4 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["different/repo"],  # Different repo IDs
        default_encoding="bfloat16",
        supported_encodings={
            "bfloat16",
        },
        pipeline_model=DummyPipelineModel,
        config=DummyLlamaArchConfig,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )
    assert arch1 != arch4

    # Test inequality with different default_encoding
    arch5 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding="q4_k",  # Different encoding
        supported_encodings={
            "bfloat16",
        },
        pipeline_model=DummyPipelineModel,
        config=DummyLlamaArchConfig,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )
    assert arch1 != arch5

    # Test inequality with different supported_encodings
    arch6 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding="bfloat16",
        supported_encodings={
            "bfloat16",
            # Missing q4_k encoding
        },
        pipeline_model=DummyPipelineModel,
        config=DummyLlamaArchConfig,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )
    assert arch1 != arch6

    # Test inequality with different pipeline_model
    arch7 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding="bfloat16",
        supported_encodings={
            "bfloat16",
        },
        pipeline_model=DUMMY_GEMMA_ARCH.pipeline_model,  # Different model
        config=DummyLlamaArchConfig,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )
    assert arch1 != arch7

    # Test inequality with different task
    arch8 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding="bfloat16",
        supported_encodings={
            "bfloat16",
        },
        pipeline_model=DummyPipelineModel,
        config=DummyLlamaArchConfig,
        task=PipelineTask.EMBEDDINGS_GENERATION,  # Different task
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )
    assert arch1 != arch8

    # Test inequality with different tokenizer
    arch9 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding="bfloat16",
        supported_encodings={
            "bfloat16",
        },
        pipeline_model=DummyPipelineModel,
        config=DummyLlamaArchConfig,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=DUMMY_GEMMA_ARCH.tokenizer,  # Different tokenizer
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )
    assert arch1 != arch9

    # Test inequality with different default_weights_format
    arch10 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding="bfloat16",
        supported_encodings={
            "bfloat16",
        },
        pipeline_model=DummyPipelineModel,
        config=DummyLlamaArchConfig,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.gguf,  # Different format
    )
    assert arch1 != arch10

    # Test inequality with different rope_type
    arch11 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding="bfloat16",
        supported_encodings={
            "bfloat16",
        },
        pipeline_model=DummyPipelineModel,
        config=DummyLlamaArchConfig,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )
    assert arch1 != arch11

    # Test inequality with different weight_adapters
    def different_adapter(x: Any) -> Any:
        return x + 1  # Different function

    arch12 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding="bfloat16",
        supported_encodings={
            "bfloat16",
        },
        pipeline_model=DummyPipelineModel,
        config=DummyLlamaArchConfig,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        weight_adapters={
            WeightsFormat.safetensors: different_adapter,  # Different weight adapters
        },
    )
    assert arch1 != arch12

    # Test inequality with different multi_gpu_supported
    arch13 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding="bfloat16",
        supported_encodings={
            "bfloat16",
        },
        pipeline_model=DummyPipelineModel,
        config=DummyLlamaArchConfig,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        multi_gpu_supported=False,  # Different multi_gpu_supported
    )
    assert arch1 != arch13

    arch14 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding="bfloat16",
        supported_encodings={
            "bfloat16",
        },
        pipeline_model=DummyPipelineModel,
        config=DummyLlamaArchConfig,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        required_arguments={"enable_prefix_caching": False},
    )
    assert arch1 != arch14

    # Test with None weight_adapters (should default to empty dict)
    arch15 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding="bfloat16",
        supported_encodings={
            "bfloat16",
        },
        pipeline_model=DummyPipelineModel,
        config=DummyLlamaArchConfig,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
    )

    arch16 = SupportedArchitecture(
        name="TestModel",
        example_repo_ids=["test/repo1", "test/repo2"],
        default_encoding="bfloat16",
        supported_encodings={
            "bfloat16",
        },
        pipeline_model=DummyPipelineModel,
        config=DummyLlamaArchConfig,
        task=PipelineTask.TEXT_GENERATION,
        tokenizer=TextTokenizer,
        context_type=TextContext,
        default_weights_format=WeightsFormat.safetensors,
        weight_adapters={},  # Empty dict
    )
    assert arch15 == arch16


def test_architecture_context_types_are_msgspec_compatible() -> None:
    """Ensure all architecture context_types work with msgspec serialization.

    See PR #74216 and PR #75135 for example bugs this test prevents.
    """
    import typing

    import msgspec

    for arch in PIPELINE_REGISTRY.all_architectures():
        context_type = arch.context_type

        # context_type must not be a Protocol (msgspec can't deserialize them)
        is_protocol = getattr(context_type, "_is_protocol", False)
        assert not is_protocol, (
            f"Architecture '{arch.name}' uses Protocol '{context_type.__name__}' "
            f"as context_type - use a concrete class instead."
        )

        # msgspec must be able to create a decoder for this type
        try:
            msgspec.msgpack.Decoder(type=context_type)
        except Exception as e:
            pytest.fail(
                f"Architecture '{arch.name}' context_type '{context_type.__name__}' "
                f"is not msgspec-compatible: {e}"
            )

        # tokenizer.new_context() return type must match context_type
        new_context_method = getattr(arch.tokenizer, "new_context", None)
        if new_context_method:
            hints = typing.get_type_hints(new_context_method)
            return_type = hints.get("return")
            if return_type and isinstance(return_type, type):
                assert issubclass(return_type, context_type), (
                    f"Architecture '{arch.name}' has context_type={context_type.__name__} "
                    f"but tokenizer.new_context() returns {return_type.__name__}."
                )


def test_registry__retrieve_chat_template_none_returns_none() -> None:
    assert _retrieve_chat_template(None) is None


@pytest.mark.parametrize(
    ("file_content", "expected"),
    [
        pytest.param("{{ messages }}", "{{ messages }}", id="plain_text"),
        pytest.param(
            json.dumps({"chat_template": "{{ messages }}"}),
            "{{ messages }}",
            id="json_with_chat_template_key",
        ),
        pytest.param(
            json.dumps({"some_other_key": "value"}),
            json.dumps({"some_other_key": "value"}),
            id="json_without_chat_template_key",
        ),
        pytest.param("not { valid json", "not { valid json", id="invalid_json"),
    ],
)
def test_registry__retrieve_chat_template_reads_file(
    tmp_path: Path, file_content: str, expected: str
) -> None:
    # Anything that isn't a JSON object with a "chat_template" key falls
    # back to the raw file content.
    template_file = tmp_path / "template.txt"
    template_file.write_text(file_content)

    assert _retrieve_chat_template(template_file) == expected


@pytest.mark.parametrize(
    ("build_path", "match"),
    [
        pytest.param(
            lambda tmp_path: tmp_path / "missing.jinja",
            "does not exist",
            id="missing",
        ),
        pytest.param(lambda tmp_path: tmp_path, "not a file", id="directory"),
    ],
)
def test_registry__retrieve_chat_template_invalid_path_raises(
    tmp_path: Path, build_path: Callable[[Path], Path], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _retrieve_chat_template(build_path(tmp_path))
