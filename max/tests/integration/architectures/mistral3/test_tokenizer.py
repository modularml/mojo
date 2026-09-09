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

"""Unit tests for Mistral3 tokenizer implementation."""

import json
from typing import Any
from unittest.mock import MagicMock, NonCallableMock, mock_open

import pytest
from max.pipelines.architectures.mistral3.tokenizer import Mistral3Tokenizer
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from pytest_mock import MockerFixture
from transformers import MistralConfig


@pytest.fixture
def mock_pipeline_config() -> NonCallableMock:
    """Create a mock PipelineConfig for testing."""
    mock_model_config = NonCallableMock(spec=MAXModelConfig)
    mock_model_config.huggingface_model_revision = None
    mock_model_config.huggingface_config = NonCallableMock(spec=MistralConfig)

    pipeline_config = NonCallableMock(spec=PipelineConfig)
    pipeline_config.model = mock_model_config
    return pipeline_config


@pytest.fixture
def mock_tokenizer_base(mocker: MockerFixture) -> None:
    """Mock the base TextTokenizer initialization."""

    def mock_init(
        instance: Any,
        model_path: str,
        pipeline_config: Any = None,
        **kwargs: Any,
    ) -> None:
        # Set the essential attributes that parent class would set
        instance.model_path = model_path
        instance.delegate = MagicMock()

    mocker.patch(
        "max.pipelines.architectures.mistral3.tokenizer.TextTokenizer.__init__",
        new=mock_init,
    )


@pytest.fixture
def sample_chat_template():  # noqa: ANN201
    """Sample chat template for testing (simplified version of real Mistral template)."""
    return {
        "chat_template": "{%- set today = strftime_now(\"%Y-%m-%d\") %}\n{{- bos_token }}\n{%- for message in messages %}\n    {%- if message['role'] == 'user' %}\n        {{- '[INST]' + message['content'] + '[/INST]' }}\n    {%- elif message['role'] == 'assistant' %}\n        {{- message['content'] + eos_token }}\n    {%- endif %}\n{%- endfor %}"
    }


def test_load_chat_template_from_cache_success(
    mocker: MockerFixture,
    mock_tokenizer_base: Any,
    mock_pipeline_config: MagicMock,
    sample_chat_template: dict[str, str],
) -> None:
    """Test successful loading of chat template from cache."""
    mock_cache = mocker.patch(
        "max.pipelines.architectures.mistral3.tokenizer.try_to_load_from_cache",
        return_value="/path/to/cached/chat_template.json",
    )
    mocker.patch(
        "builtins.open",
        mock_open(read_data=json.dumps(sample_chat_template)),
    )
    mocker.patch("pathlib.Path.exists", return_value=True)

    tokenizer = Mistral3Tokenizer("test/model", mock_pipeline_config)

    # Verify cache was queried
    mock_cache.assert_called_once_with(
        repo_id="test/model",
        filename="chat_template.json",
        revision="main",
    )

    # Verify template was set
    assert (
        tokenizer.delegate.chat_template
        == sample_chat_template["chat_template"]
    )


def test_load_chat_template_download_fallback_success(
    mocker: MockerFixture,
    mock_tokenizer_base: Any,
    mock_pipeline_config: MagicMock,
    sample_chat_template: dict[str, str],
) -> None:
    """Test successful downloading when not in cache."""
    mocker.patch(
        "max.pipelines.architectures.mistral3.tokenizer.try_to_load_from_cache",
        return_value=None,
    )
    mock_download = mocker.patch(
        "max.pipelines.architectures.mistral3.tokenizer.huggingface_hub.hf_hub_download",
        return_value="/path/to/downloaded/chat_template.json",
    )
    mocker.patch(
        "builtins.open",
        mock_open(read_data=json.dumps(sample_chat_template)),
    )
    mocker.patch("pathlib.Path.exists", return_value=True)

    tokenizer = Mistral3Tokenizer("test/model", mock_pipeline_config)

    # Verify download was attempted
    mock_download.assert_called_once_with(
        repo_id="test/model",
        filename="chat_template.json",
        revision="main",
    )

    # Verify template was set
    assert (
        tokenizer.delegate.chat_template
        == sample_chat_template["chat_template"]
    )


def test_load_chat_template_with_revision(
    mocker: MockerFixture,
    mock_tokenizer_base: Any,
    mock_pipeline_config: MagicMock,
    sample_chat_template: dict[str, str],
) -> None:
    """Test loading with custom revision."""
    mock_cache = mocker.patch(
        "max.pipelines.architectures.mistral3.tokenizer.try_to_load_from_cache",
        return_value="/path/to/cached/chat_template.json",
    )
    mocker.patch(
        "builtins.open",
        mock_open(read_data=json.dumps(sample_chat_template)),
    )
    mocker.patch("pathlib.Path.exists", return_value=True)

    Mistral3Tokenizer("test/model", mock_pipeline_config, revision="v1.0")

    # Verify cache was queried with correct revision
    mock_cache.assert_called_once_with(
        repo_id="test/model",
        filename="chat_template.json",
        revision="v1.0",
    )


def test_load_chat_template_with_pipeline_config_revision(
    mocker: MockerFixture,
    mock_tokenizer_base: Any,
    sample_chat_template: dict[str, str],
) -> None:
    """Test loading with revision from pipeline config."""
    mock_model_config = NonCallableMock(spec=MAXModelConfig)
    mock_model_config.huggingface_model_revision = "config-revision"

    pipeline_config = NonCallableMock(spec=PipelineConfig)
    pipeline_config.model = mock_model_config

    mock_cache = mocker.patch(
        "max.pipelines.architectures.mistral3.tokenizer.try_to_load_from_cache",
        return_value="/path/to/cached/chat_template.json",
    )
    mocker.patch(
        "builtins.open",
        mock_open(read_data=json.dumps(sample_chat_template)),
    )
    mocker.patch("pathlib.Path.exists", return_value=True)

    Mistral3Tokenizer("test/model", pipeline_config)

    # Verify cache was queried with config revision
    mock_cache.assert_called_once_with(
        repo_id="test/model",
        filename="chat_template.json",
        revision="config-revision",
    )


def test_load_chat_template_revision_precedence(
    mocker: MockerFixture,
    mock_tokenizer_base: Any,
    sample_chat_template: dict[str, str],
) -> None:
    """Test that explicit revision takes precedence over config revision."""
    mock_model_config = NonCallableMock(spec=MAXModelConfig)
    mock_model_config.huggingface_model_revision = "config-revision"

    pipeline_config = NonCallableMock(spec=PipelineConfig)
    pipeline_config.model = mock_model_config

    mock_cache = mocker.patch(
        "max.pipelines.architectures.mistral3.tokenizer.try_to_load_from_cache",
        return_value="/path/to/cached/chat_template.json",
    )
    mocker.patch(
        "builtins.open",
        mock_open(read_data=json.dumps(sample_chat_template)),
    )
    mocker.patch("pathlib.Path.exists", return_value=True)

    Mistral3Tokenizer(
        "test/model",
        pipeline_config,
        revision="explicit-revision",
    )

    # Verify explicit revision is used
    mock_cache.assert_called_once_with(
        repo_id="test/model",
        filename="chat_template.json",
        revision="explicit-revision",
    )
