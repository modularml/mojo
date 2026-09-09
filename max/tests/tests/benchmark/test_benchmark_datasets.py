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

"""Unit tests for benchmark_shared.datasets module."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import msgspec
import pytest
from huggingface_hub import errors as hf_hub_errors
from max.benchmark.benchmark_shared.datasets import (
    DATASET_REGISTRY,
    ArtificialAnalysisBenchmarkDataset,
    ArxivSummarizationBenchmarkDataset,
    AxolotlBenchmarkDataset,
    BenchmarkDataset,
    CodeDebugBenchmarkDataset,
    DatasetRegistryEntry,
    HuggingFaceBenchmarkDataset,
    InstructCoderBenchmarkDataset,
    LocalBenchmarkDataset,
    LocalImageBenchmarkDataset,
    NemotronOpenCodeBenchmarkDataset,
    ObfuscatedConversationsBenchmarkDataset,
    PixelGenerationSampledRequest,
    RandomBenchmarkDataset,
    SampledRequest,
    SharedContext,
    ShareGPTBenchmarkDataset,
    SonnetBenchmarkDataset,
    SyntheticPixelBenchmarkDataset,
    VisionArenaBenchmarkDataset,
)
from max.benchmark.benchmark_shared.datasets._hf_download import (
    hf_hub_download_with_retry,
)
from max.benchmark.benchmark_shared.datasets._tokenizer_pool import (
    TokenizerPool,
    _init_encoder,
)
from max.benchmark.benchmark_shared.datasets.chat_judge import (
    ChatJudgeBenchmarkDataset,
)

# Import the module under test
from max.benchmark.benchmark_shared.datasets.multiturn_distribution_fit import (
    build_chat_samples_from_user_text_pool,
    resolve_constant_delay_ms,
)
from max.benchmark.benchmark_shared.datasets.nemotron_opencode import (
    NEMOTRON_OPENCODE_REPO_ID,
    AnthropicTool,
    _anthropic_tool_to_openai,
)
from PIL import Image
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class _FakeTokenizer:
    """Picklable stand-in for `PreTrainedTokenizerBase` used by unit tests.

    `_fake_loader` constructs one per spawn worker; the parent uses an
    instance of this class directly. Behavior is deterministic so tests
    can assert on outputs (no Mock call-count plumbing).
    """

    name_or_path = "_fake_"
    vocab_size = 1000
    unk_token_id = None
    all_special_ids: frozenset[int] = frozenset({0, 1, 2})

    def __init__(self, model_max_length: int = 4096) -> None:
        self.model_max_length = model_max_length

    def encode(
        self, text: str, add_special_tokens: bool = False, **_: object
    ) -> list[int]:
        # Length-aware so multiturn-fit tests that scale messages to a
        # target token count round-trip through encode→decode→encode and
        # see a meaningful encoded length.
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
    return _FakeTokenizer(model_max_length=model_max_length or 4096)


def _raising_loader(
    name_or_path: str,
    model_max_length: int | None,
    trust_remote_code: bool,
    revision: str | None,
) -> _FakeTokenizer:
    raise OSError("simulated Hub-offline tokenizer load failure")


def test_init_encoder_exits_quietly_on_loader_failure() -> None:
    """A worker whose tokenizer load fails exits via `SystemExit` with a
    logged warning, instead of letting the raw exception escape the pool
    initializer (which `BaseProcess._bootstrap` would otherwise print as an
    unhandled-exception traceback per crashed worker -- see MXSERV-373)."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    log_queue: mp.Queue[logging.LogRecord] = mp.get_context("spawn").Queue(-1)
    try:
        with pytest.raises(SystemExit):
            _init_encoder(
                "fake-model",
                None,
                False,
                None,
                None,
                _raising_loader,
                log_queue,
                logging.INFO,
            )
        record = log_queue.get(timeout=5)
        assert record.levelno == logging.WARNING
        assert "fake-model" in record.getMessage()
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_dataset_registry_structure() -> None:
    """Test that the registry has the expected structure."""
    assert isinstance(DATASET_REGISTRY, dict)
    assert len(DATASET_REGISTRY) > 0

    for dataset_name, dataset_info in DATASET_REGISTRY.items():
        assert isinstance(dataset_name, str)
        assert isinstance(dataset_info, DatasetRegistryEntry)
        assert dataset_info.class_name is not None
        assert dataset_info.has_multiturn_chat_support is not None


def test_dataset_registry_contents() -> None:
    """Test that the registry contains expected datasets."""
    expected_datasets = {
        "agentic-code",
        "artificial-analysis",
        "arxiv-summarization",
        "chat-judge",
        "instruct-coder",
        "nemotron-opencode",
        "sharegpt",
        "code_debug",
        "random",
        "synthetic",
        "synthetic-pixel",
        "sonnet",
        "vision-arena",
        "axolotl",
        "batch-job",
        "obfuscated-conversations",
        "local-image",
    }
    assert set(DATASET_REGISTRY.keys()) == expected_datasets


def test_dataset_registry_multiturn_support_mapping() -> None:
    """Test that multiturn support is correctly configured."""
    expected_multiturn = {
        "agentic-code": True,
        "arxiv-summarization": False,
        "code_debug": True,
        "instruct-coder": True,
        "nemotron-opencode": True,
        "random": True,
        "synthetic": True,
        "sharegpt": False,
        "sonnet": False,
        "vision-arena": False,
        "axolotl": False,
        "batch-job": False,
        "synthetic-pixel": False,
        "local-image": False,
    }

    for dataset_name, expected_support in expected_multiturn.items():
        actual_support = DATASET_REGISTRY[
            dataset_name
        ].has_multiturn_chat_support
        assert actual_support == expected_support


def test_dataset_registry_class_names_exist() -> None:
    """Test that all referenced class names exist in globals."""
    from max.benchmark.benchmark_shared import datasets

    for dataset_info in DATASET_REGISTRY.values():
        class_name = dataset_info.class_name
        assert hasattr(datasets, class_name), (
            f"Class {class_name} not found in benchmark_shared.datasets"
        )


@patch("os.path.exists")
def test_benchmark_dataset_from_flags_with_dataset_name(
    mock_exists: Mock,
) -> None:
    """Test creating dataset instances using dataset_name."""
    mock_exists.return_value = True  # Mock any downloaded paths exist

    with patch.object(ShareGPTBenchmarkDataset, "fetch"):
        dataset = BenchmarkDataset.from_flags(dataset_name="sharegpt")

        assert isinstance(dataset, ShareGPTBenchmarkDataset)
        assert dataset.dataset_name == "sharegpt"
        assert not dataset.has_multiturn_chat_support


@patch("os.path.exists")
def test_benchmark_dataset_from_flags_with_dataset_path(
    mock_exists: Mock,
) -> None:
    """Test creating dataset instances using dataset_path."""
    mock_exists.return_value = True  # Mock path exists

    dataset = BenchmarkDataset.from_flags(
        dataset_name="random", dataset_path="/path/to/dataset.json"
    )

    assert isinstance(dataset, RandomBenchmarkDataset)
    assert dataset.dataset_name == "random"
    assert dataset.dataset_path == "/path/to/dataset.json"
    assert dataset.has_multiturn_chat_support
    mock_exists.assert_called_once_with("/path/to/dataset.json")


def test_benchmark_dataset_from_flags_no_parameters() -> None:
    """Test that providing neither dataset_name nor dataset_path raises error."""
    with pytest.raises(
        ValueError, match="Either dataset_name or dataset_path must be provided"
    ):
        BenchmarkDataset.from_flags()


@patch("os.path.exists")
def test_benchmark_dataset_from_flags_no_dataset_name(
    mock_exists: Mock,
) -> None:
    """Test that providing only dataset_path without dataset_name raises error."""
    mock_exists.return_value = True  # Mock path exists

    with pytest.raises(ValueError) as exc_info:
        BenchmarkDataset.from_flags(dataset_path="/path/to/dataset.json")

    mock_exists.assert_called_once_with("/path/to/dataset.json")
    assert "dataset_name is required" in str(exc_info.value)


@patch("os.path.exists")
def test_benchmark_dataset_from_flags_nonexistent_path(
    mock_exists: Mock,
) -> None:
    """Test that providing a nonexistent dataset_path raises error."""
    mock_exists.return_value = False  # Mock path doesn't exist

    with pytest.raises(ValueError) as exc_info:
        BenchmarkDataset.from_flags(
            dataset_name="random", dataset_path="/nonexistent/path.json"
        )

    assert "Dataset path /nonexistent/path.json does not exist" in str(
        exc_info.value
    )
    mock_exists.assert_called_once_with("/nonexistent/path.json")


def test_benchmark_dataset_from_flags_unknown_dataset() -> None:
    """Test that unknown dataset names raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        BenchmarkDataset.from_flags(dataset_name="unknown_dataset")

    assert "Unknown dataset: unknown_dataset" in str(exc_info.value)
    assert "Available datasets:" in str(exc_info.value)


def test_benchmark_dataset_get_dataset_class() -> None:
    """Test the _get_dataset_class method."""
    dataset_class = BenchmarkDataset._get_dataset_class("sharegpt")
    assert dataset_class == ShareGPTBenchmarkDataset

    dataset_class = BenchmarkDataset._get_dataset_class("code_debug")
    assert dataset_class == CodeDebugBenchmarkDataset


def test_benchmark_dataset_get_dataset_class_unknown() -> None:
    """Test _get_dataset_class with unknown dataset name."""
    with pytest.raises(ValueError) as exc_info:
        BenchmarkDataset._get_dataset_class("unknown_dataset")

    assert "Unknown dataset: unknown_dataset" in str(exc_info.value)


@patch("os.path.exists")
def test_benchmark_dataset_str_with_dataset_name(mock_exists: Mock) -> None:
    """Test __str__ method with dataset_name."""
    mock_exists.return_value = True  # Mock any downloaded paths exist

    with patch.object(ShareGPTBenchmarkDataset, "fetch"):
        dataset = BenchmarkDataset.from_flags(dataset_name="sharegpt")
        str_repr = str(dataset)
        assert "sharegpt" in str_repr
        assert "ShareGPTBenchmarkDataset" in str_repr


@patch("os.path.exists")
def test_benchmark_dataset_str_with_dataset_path(mock_exists: Mock) -> None:
    """Test __str__ method with dataset_path."""
    mock_exists.return_value = True  # Mock path exists

    dataset = BenchmarkDataset.from_flags(
        dataset_name="random", dataset_path="/path/to/dataset.json"
    )
    str_repr = str(dataset)
    assert "local_dataset_at_/path/to/dataset.json" in str_repr
    assert "RandomBenchmarkDataset" in str_repr


@patch("os.path.exists")
def test_benchmark_dataset_repr(mock_exists: Mock) -> None:
    """Test __repr__ method."""
    mock_exists.return_value = True  # Mock path exists

    dataset = BenchmarkDataset.from_flags(
        dataset_name="random", dataset_path="/path/to/dataset.json"
    )
    repr_str = repr(dataset)
    assert "RandomBenchmarkDataset" in repr_str
    assert "dataset_name='random'" in repr_str
    assert "dataset_path='/path/to/dataset.json'" in repr_str
    assert "has_multiturn_chat_support=True" in repr_str


@patch("os.path.exists")
@patch(
    "max.benchmark.benchmark_shared.datasets.code_debug.hf_hub_download_with_retry"
)
def test_code_debug_from_flags(mock_download: Mock, mock_exists: Mock) -> None:
    """Test fetching code_debug dataset from HuggingFace Hub."""
    mock_download.return_value = "/path/to/downloaded/file.jsonl"
    mock_exists.return_value = True  # Mock downloaded file exists

    dataset = BenchmarkDataset.from_flags(dataset_name="code_debug")

    assert dataset.dataset_path == "/path/to/downloaded/file.jsonl"
    mock_download.assert_called_once_with(
        repo_id="xinrongzhang2022/InfiniteBench",
        filename="code_debug.jsonl",
        repo_type="dataset",
    )


@patch(
    "max.benchmark.benchmark_shared.datasets.code_debug.hf_hub_download_with_retry"
)
def test_code_debug_from_flags_unknown(mock_download: Mock) -> None:
    """Test fetching unknown dataset raises ValueError."""
    mock_download.return_value = "/tmp/fake_code_debug.jsonl"
    dataset = BenchmarkDataset.from_flags(dataset_name="code_debug")
    # CodeDebugBenchmarkDataset now inherits from HuggingFaceBenchmarkDataset
    # and will fetch from HF by default, so we test the fetch method directly
    with patch.object(CodeDebugBenchmarkDataset, "fetch") as mock_fetch:
        mock_fetch.side_effect = ValueError(
            "Unknown dataset for CodeDebugBenchmarkDataset: unknown_dataset"
        )
        with pytest.raises(ValueError, match="Unknown dataset"):
            dataset.fetch()


@patch("os.path.exists")
@patch(
    "max.benchmark.benchmark_shared.datasets.sharegpt.hf_hub_download_with_retry"
)
def test_sharegpt_from_flags(mock_download: Mock, mock_exists: Mock) -> None:
    """Test fetching ShareGPT dataset from HuggingFace Hub."""
    mock_download.return_value = "/path/to/downloaded/file.json"
    mock_exists.return_value = True  # Mock downloaded file exists

    dataset = BenchmarkDataset.from_flags(dataset_name="sharegpt")

    assert dataset.dataset_path == "/path/to/downloaded/file.json"
    mock_download.assert_called_once_with(
        repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
        filename="ShareGPT_V3_unfiltered_cleaned_split.json",
        repo_type="dataset",
    )


def test_random_from_flags() -> None:
    """Test that RandomBenchmarkDataset HF fetching being essentially a no-op."""
    # This shouldn't raise an error because random dataset is not fetched from HF.
    dataset = BenchmarkDataset.from_flags(dataset_name="random")
    assert isinstance(dataset, RandomBenchmarkDataset)
    assert dataset.dataset_name == "random"
    assert dataset.dataset_path is None


def test_random_sample_requests() -> None:
    """Test sampling random requests."""
    tok = _FakeTokenizer(model_max_length=50)
    dataset = BenchmarkDataset.from_flags(dataset_name="random")
    assert isinstance(dataset, RandomBenchmarkDataset)

    with TokenizerPool(tok, loader=_fake_loader) as pool:
        samples = dataset.sample_requests(
            num_requests=2,
            tokenizer=tok,
            pool=pool,
            input_len="N(50, 5)",
            output_len="N(20, 2)",
            sys_prompt_ratio=0.1,
            max_num_unique_sys_prompt=1,
        )

    assert len(samples.requests) == 2
    for request in samples.requests:
        assert isinstance(request, SampledRequest)


def test_shared_contexts_empty_when_no_sys_prompt() -> None:
    """shared_contexts is empty when sys_prompt_ratio is 0."""
    tok = _FakeTokenizer()
    dataset = BenchmarkDataset.from_flags(dataset_name="random")
    assert isinstance(dataset, RandomBenchmarkDataset)

    with TokenizerPool(tok, loader=_fake_loader) as pool:
        samples = dataset.sample_requests(
            num_requests=5,
            tokenizer=tok,
            pool=pool,
            input_len="50",
            output_len="10",
            sys_prompt_ratio=0.0,
            max_num_unique_sys_prompt=1,
        )

    assert samples.shared_contexts == []


def test_shared_contexts_one_entry_per_unique_idx() -> None:
    """shared_contexts has exactly one SharedContext per unique sys_prompt_idx."""
    tok = _FakeTokenizer()
    dataset = BenchmarkDataset.from_flags(dataset_name="random")
    assert isinstance(dataset, RandomBenchmarkDataset)

    with TokenizerPool(tok, loader=_fake_loader) as pool:
        samples = dataset.sample_requests(
            num_requests=10,
            tokenizer=tok,
            pool=pool,
            input_len="50",
            output_len="10",
            sys_prompt_ratio=0.3,
            max_num_unique_sys_prompt=1,
        )

    assert len(samples.shared_contexts) == 1
    assert isinstance(samples.shared_contexts[0], SharedContext)


def test_shared_contexts_at_most_max_unique() -> None:
    """shared_contexts has at most max_num_unique_sys_prompt entries."""
    tok = _FakeTokenizer()
    dataset = BenchmarkDataset.from_flags(dataset_name="random")
    assert isinstance(dataset, RandomBenchmarkDataset)

    max_unique = 3
    with TokenizerPool(tok, loader=_fake_loader) as pool:
        samples = dataset.sample_requests(
            num_requests=30,
            tokenizer=tok,
            pool=pool,
            input_len="50",
            output_len="10",
            sys_prompt_ratio=0.3,
            max_num_unique_sys_prompt=max_unique,
        )

    assert len(samples.shared_contexts) <= max_unique
    for entry in samples.shared_contexts:
        assert isinstance(entry, SharedContext)
        assert entry.num_tokens > 0


@patch("os.path.exists")
def test_sonnet_from_flags_local_mode(mock_exists: Mock) -> None:
    """Test that SonnetBenchmarkDataset works in local mode."""
    mock_exists.return_value = True

    dataset = BenchmarkDataset.from_flags(
        dataset_name="sonnet", dataset_path="/path/to/sonnet.txt"
    )

    assert isinstance(dataset, SonnetBenchmarkDataset)
    assert dataset.dataset_name == "sonnet"
    assert dataset.dataset_path == "/path/to/sonnet.txt"
    assert dataset.dataset_mode == "local"


@patch("os.path.exists")
def test_vision_arena_from_flags_local_mode(mock_exists: Mock) -> None:
    """Test that VisionArenaBenchmarkDataset works in local mode."""
    mock_exists.return_value = True

    dataset = BenchmarkDataset.from_flags(
        dataset_name="vision-arena", dataset_path="/path/to/vision_arena.json"
    )

    assert isinstance(dataset, VisionArenaBenchmarkDataset)
    assert dataset.dataset_name == "vision-arena"
    assert dataset.dataset_path == "/path/to/vision_arena.json"
    assert dataset.dataset_mode == "local"


@patch("os.path.exists")
def test_axolotl_from_flags_local_mode(mock_exists: Mock) -> None:
    """Test that AxolotlBenchmarkDataset works in local mode."""
    mock_exists.return_value = True

    dataset = BenchmarkDataset.from_flags(
        dataset_name="axolotl", dataset_path="/path/to/axolotl.json"
    )

    assert isinstance(dataset, AxolotlBenchmarkDataset)
    assert dataset.dataset_name == "axolotl"
    assert dataset.dataset_path == "/path/to/axolotl.json"
    assert dataset.dataset_mode == "local"


@patch("os.path.exists")
def test_arxiv_summarization_from_flags_local_mode(mock_exists: Mock) -> None:
    """Test that ArxivSummarizationBenchmarkDataset works in local mode."""
    mock_exists.return_value = True

    dataset = BenchmarkDataset.from_flags(
        dataset_name="arxiv-summarization", dataset_path="/path/to/arxiv.json"
    )

    assert isinstance(dataset, ArxivSummarizationBenchmarkDataset)
    assert dataset.dataset_name == "arxiv-summarization"
    assert dataset.dataset_path == "/path/to/arxiv.json"
    assert dataset.dataset_mode == "local"


def test_vision_arena_from_flags_unknown() -> None:
    """Test fetching unknown dataset raises ValueError."""
    with pytest.raises(ValueError, match="Unknown dataset"):
        BenchmarkDataset.from_flags(dataset_name="vision-arena-unknown")


@patch("os.path.exists")
def test_obfuscated_conversations_from_flags_local_mode(
    mock_exists: Mock,
) -> None:
    """Test that ObfuscatedConversationsBenchmarkDataset works in local mode."""
    mock_exists.return_value = True

    dataset = BenchmarkDataset.from_flags(
        dataset_name="obfuscated-conversations",
        dataset_path="/path/to/obfuscated.jsonl",
    )

    assert isinstance(dataset, ObfuscatedConversationsBenchmarkDataset)
    assert dataset.dataset_name == "obfuscated-conversations"
    assert dataset.dataset_path == "/path/to/obfuscated.jsonl"
    assert dataset.dataset_mode == "local"


@patch("os.path.exists")
def test_obfuscated_conversations_missing_seed(mock_exists: Mock) -> None:
    """Test that ObfuscatedConversationsBenchmarkDataset requires seed parameter."""
    mock_exists.return_value = True

    # Mock tokenizer
    mock_tokenizer = Mock(spec=PreTrainedTokenizerBase)
    mock_tokenizer.vocab_size = 1000
    mock_tokenizer.all_special_ids = {0, 1, 2}
    mock_tokenizer.encode.return_value = [100]
    mock_tokenizer.decode.return_value = "random text"
    mock_tokenizer.return_value.input_ids = [100, 101, 102]

    dataset = BenchmarkDataset.from_flags(
        dataset_name="obfuscated-conversations",
        dataset_path="/path/to/obfuscated.jsonl",
    )
    assert isinstance(dataset, ObfuscatedConversationsBenchmarkDataset)

    # Test that missing seed parameter raises ValueError
    with pytest.raises(
        ValueError,
        match="seed is required for ObfuscatedConversationsBenchmarkDataset",
    ):
        dataset.sample_requests(
            num_requests=2,
            tokenizer=mock_tokenizer,
            output_lengths=[10, 20],
        )


def test_obfuscated_conversations_with_seed() -> None:
    """Test that ObfuscatedConversationsBenchmarkDataset works with seed parameter."""
    # Mock tokenizer
    mock_tokenizer = Mock(spec=PreTrainedTokenizerBase)
    mock_tokenizer.vocab_size = 1000
    mock_tokenizer.all_special_ids = {0, 1, 2}
    mock_tokenizer.encode.return_value = [100]
    mock_tokenizer.decode.return_value = "random text"
    mock_tokenizer.return_value.input_ids = [100, 101, 102]

    # Test that providing seed parameter works
    with (
        patch(
            "builtins.open",
            mock_open(
                read_data='{"timestamp": "2023-01-01", "conversation_id": "conv1", "messages": "test message"}\n{"timestamp": "2023-01-02", "conversation_id": "conv2", "messages": "test message 2"}\n'
            ),
        ),
        patch("os.path.exists", return_value=True),
    ):
        dataset = BenchmarkDataset.from_flags(
            dataset_name="obfuscated-conversations",
            dataset_path="/path/to/test/dataset.jsonl",
        )
        assert isinstance(dataset, ObfuscatedConversationsBenchmarkDataset)
        samples = dataset.sample_requests(
            num_requests=2,
            tokenizer=mock_tokenizer,
            output_lengths=[10, 20],
            seed=42,
        )
        assert len(samples.requests) == 2
        for request in samples.requests:
            assert isinstance(request, SampledRequest)


# Tests for new base classes
def test_local_benchmark_dataset_base_class() -> None:
    """Test LocalBenchmarkDataset base class behavior."""
    # Test that LocalBenchmarkDataset has LOCAL mode by default
    assert LocalBenchmarkDataset.dataset_mode == "local"

    # Test that it sets default dataset_path and validates it exists
    dataset = AxolotlBenchmarkDataset()
    dataset.dataset_name = "test"

    # Test that it sets a default path (may or may not exist in test environment)
    dataset.fetch()
    assert dataset.dataset_path is not None
    assert "axolotl_dummy.json" in dataset.dataset_path

    # Test with explicit valid dataset_path
    with patch("os.path.exists", return_value=True):
        dataset.dataset_path = "/path/to/dataset.json"
        dataset.fetch()  # Should not raise


@patch(
    "max.benchmark.benchmark_shared.datasets.code_debug.hf_hub_download_with_retry"
)
def test_huggingface_benchmark_dataset_base_class(mock_download: Mock) -> None:
    """Test HuggingFaceBenchmarkDataset base class behavior."""
    mock_download.return_value = "/tmp/fake_code_debug.jsonl"
    # Test that HuggingFaceBenchmarkDataset has HUGGINGFACE mode by default
    assert HuggingFaceBenchmarkDataset.dataset_mode == "huggingface"

    # Test that fetch is a no-op by default using a concrete implementation
    dataset = CodeDebugBenchmarkDataset()
    dataset.fetch()  # Should not raise


def test_dataset_inheritance_hierarchy() -> None:
    """Test that datasets inherit from the correct base classes."""
    # Test local datasets
    assert issubclass(AxolotlBenchmarkDataset, LocalBenchmarkDataset)
    assert issubclass(SonnetBenchmarkDataset, LocalBenchmarkDataset)
    assert issubclass(RandomBenchmarkDataset, LocalBenchmarkDataset)
    assert issubclass(VisionArenaBenchmarkDataset, LocalBenchmarkDataset)
    assert issubclass(ArxivSummarizationBenchmarkDataset, LocalBenchmarkDataset)
    assert issubclass(
        ObfuscatedConversationsBenchmarkDataset, LocalBenchmarkDataset
    )

    # Test HuggingFace datasets
    assert issubclass(CodeDebugBenchmarkDataset, HuggingFaceBenchmarkDataset)
    assert issubclass(ShareGPTBenchmarkDataset, HuggingFaceBenchmarkDataset)

    # Test that all datasets inherit from BenchmarkDataset
    assert issubclass(LocalBenchmarkDataset, BenchmarkDataset)
    assert issubclass(HuggingFaceBenchmarkDataset, BenchmarkDataset)
    assert issubclass(AxolotlBenchmarkDataset, BenchmarkDataset)
    assert issubclass(CodeDebugBenchmarkDataset, BenchmarkDataset)


def test_local_datasets_require_dataset_path() -> None:
    """Test that local datasets set default dataset_path and validate it exists."""
    # Test AxolotlBenchmarkDataset
    axolotl_dataset = AxolotlBenchmarkDataset()
    axolotl_dataset.dataset_name = "axolotl"

    # Test that it sets a default path
    axolotl_dataset.fetch()
    assert axolotl_dataset.dataset_path is not None
    assert "axolotl_dummy.json" in axolotl_dataset.dataset_path

    # Test SonnetBenchmarkDataset
    sonnet_dataset = SonnetBenchmarkDataset()
    sonnet_dataset.dataset_name = "sonnet"

    # Test that it sets a default path
    sonnet_dataset.fetch()
    assert sonnet_dataset.dataset_path is not None
    assert "sonnet_4x.txt" in sonnet_dataset.dataset_path


def test_huggingface_datasets_fetch_behavior() -> None:
    """Test that HuggingFace datasets fetch from HF when used directly."""
    # Test CodeDebugBenchmarkDataset
    with patch(
        "max.benchmark.benchmark_shared.datasets.code_debug.hf_hub_download_with_retry"
    ) as mock_download:
        mock_download.return_value = "/path/to/downloaded/file.jsonl"

        code_debug_dataset = CodeDebugBenchmarkDataset()
        code_debug_dataset.dataset_name = "code_debug"
        code_debug_dataset.fetch()

        assert (
            code_debug_dataset.dataset_path == "/path/to/downloaded/file.jsonl"
        )
        mock_download.assert_called_once_with(
            repo_id="xinrongzhang2022/InfiniteBench",
            filename="code_debug.jsonl",
            repo_type="dataset",
        )

    # Test ShareGPTBenchmarkDataset
    with patch(
        "max.benchmark.benchmark_shared.datasets.sharegpt.hf_hub_download_with_retry"
    ) as mock_download:
        mock_download.return_value = "/path/to/downloaded/file.json"

        sharegpt_dataset = ShareGPTBenchmarkDataset()
        sharegpt_dataset.dataset_name = "sharegpt"
        sharegpt_dataset.fetch()

        assert sharegpt_dataset.dataset_path == "/path/to/downloaded/file.json"
        mock_download.assert_called_once_with(
            repo_id="anon8231489123/ShareGPT_Vicuna_unfiltered",
            filename="ShareGPT_V3_unfiltered_cleaned_split.json",
            repo_type="dataset",
        )


def test_local_datasets_with_default_files() -> None:
    """Test that local datasets work with default files when using from_flags."""
    # Test Axolotl with default file
    with patch("os.path.exists", return_value=True):
        dataset = BenchmarkDataset.from_flags(dataset_name="axolotl")
        assert isinstance(dataset, AxolotlBenchmarkDataset)
        assert dataset.dataset_name == "axolotl"
        assert dataset.dataset_mode == "local"
        # Should have set the default path
        assert dataset.dataset_path is not None
        assert "axolotl_dummy.json" in dataset.dataset_path

    # Test Sonnet with default file
    with patch("os.path.exists", return_value=True):
        dataset = BenchmarkDataset.from_flags(dataset_name="sonnet")
        assert isinstance(dataset, SonnetBenchmarkDataset)
        assert dataset.dataset_name == "sonnet"
        assert dataset.dataset_mode == "local"
        # Should have set the default path
        assert dataset.dataset_path is not None
        assert "sonnet_4x.txt" in dataset.dataset_path


def _write_test_image(image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color="red").save(image_path)


def test_image_edit_dataset_sample_requests(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "sample.png"
    _write_test_image(image_path)
    dataset_path = tmp_path / "image_edit.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "prompt": "Turn this into watercolor",
                "image_path": "images/sample.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = BenchmarkDataset.from_flags(
        dataset_name="local-image",
        dataset_path=str(dataset_path),
    )
    assert isinstance(dataset, LocalImageBenchmarkDataset)

    samples = dataset.sample_requests(
        num_requests=1,
        tokenizer=None,
        image_width=1024,
        image_steps=28,
        image_guidance_scale=3.5,
    )
    request = samples.requests[0]
    assert isinstance(request, PixelGenerationSampledRequest)
    assert request.prompt_formatted == "Turn this into watercolor"
    assert request.input_image_paths == [str(image_path.resolve())]
    assert request.image_options is not None
    assert request.image_options.width == 1024
    assert request.image_options.steps == 28
    assert request.image_options.guidance_scale == 3.5


def test_image_edit_dataset_forwards_num_frames(tmp_path: Path) -> None:
    """image-to-video reuses the local-image dataset and threads num_frames."""
    image_path = tmp_path / "images" / "sample.png"
    _write_test_image(image_path)
    dataset_path = tmp_path / "image_edit.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "prompt": "Pan across the scene",
                "image_path": "images/sample.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = BenchmarkDataset.from_flags(
        dataset_name="local-image",
        dataset_path=str(dataset_path),
    )
    assert isinstance(dataset, LocalImageBenchmarkDataset)

    samples = dataset.sample_requests(
        num_requests=1,
        tokenizer=None,
        image_width=832,
        image_height=480,
        num_frames=81,
    )
    request = samples.requests[0]
    assert isinstance(request, PixelGenerationSampledRequest)
    assert request.input_image_paths == [str(image_path.resolve())]
    assert request.image_options is not None
    assert request.image_options.num_frames == 81


def test_image_edit_dataset_invalid_jsonl(tmp_path: Path) -> None:
    dataset_path = tmp_path / "bad.jsonl"
    dataset_path.write_text("{not-json}\n", encoding="utf-8")
    dataset = BenchmarkDataset.from_flags(
        dataset_name="local-image",
        dataset_path=str(dataset_path),
    )

    with pytest.raises(ValueError, match="Invalid JSON"):
        dataset.sample_requests(num_requests=1, tokenizer=None)


def test_synthetic_pixel_dataset_sample_requests_for_image_to_image() -> None:
    dataset = BenchmarkDataset.from_flags(dataset_name="synthetic-pixel")
    assert isinstance(dataset, SyntheticPixelBenchmarkDataset)

    samples = dataset.sample_requests(
        num_requests=2,
        tokenizer=None,
        benchmark_task="image-to-image",
        image_width=640,
        image_height=832,
        image_steps=18,
        image_guidance_scale=3.0,
    )

    assert len(samples.requests) == 2
    for request in samples.requests:
        assert isinstance(request, PixelGenerationSampledRequest)
        assert request.prompt_formatted.startswith("Random prompt")
        assert len(request.input_image_paths) == 1
        assert Path(request.input_image_paths[0]).exists()
        assert request.image_options is not None
        assert request.image_options.width == 640
        assert request.image_options.height == 832
        assert request.image_options.steps == 18
        assert request.image_options.guidance_scale == 3.0


def test_synthetic_pixel_dataset_sample_requests_for_image_to_video() -> None:
    dataset = BenchmarkDataset.from_flags(dataset_name="synthetic-pixel")
    assert isinstance(dataset, SyntheticPixelBenchmarkDataset)

    samples = dataset.sample_requests(
        num_requests=2,
        tokenizer=None,
        benchmark_task="image-to-video",
        image_width=832,
        image_height=480,
        num_frames=81,
    )

    assert len(samples.requests) == 2
    for request in samples.requests:
        assert isinstance(request, PixelGenerationSampledRequest)
        assert request.prompt_formatted.startswith("Random prompt")
        # image-to-video still attaches an input image, like image-to-image.
        assert len(request.input_image_paths) == 1
        assert Path(request.input_image_paths[0]).exists()
        assert request.image_options is not None
        assert request.image_options.num_frames == 81


def test_random_multiturn_emits_zero_prefix_turns() -> None:
    """gen_multiturn_random_requests always emits prefix_turns=0; the runner
    owns warmup prefix-turn assignment via _pick_warmup_population."""
    tok = _FakeTokenizer()
    dataset = BenchmarkDataset.from_flags(dataset_name="random")
    assert isinstance(dataset, RandomBenchmarkDataset)
    with TokenizerPool(tok, loader=_fake_loader) as pool:
        samples = dataset.gen_multiturn_random_requests(
            input_len=32,
            output_len=16,
            num_chat_sessions=20,
            num_turns=3,
            delay_between_chat_turns=500,
            pool=pool,
            sys_prompt_ratio=0.0,
            max_num_unique_sys_prompt=1,
        )

    assert all(s.prefix_turns == 0 for s in samples.chat_sessions)


def test_resolve_constant_delay_ms() -> None:
    assert resolve_constant_delay_ms(None) is None
    assert resolve_constant_delay_ms(12) == 12.0
    assert resolve_constant_delay_ms("0") == 0.0


@patch.object(InstructCoderBenchmarkDataset, "_load_pairs")
def test_instruct_coder_multiturn_fit_distributions(
    mock_load_pairs: Mock,
) -> None:
    """``fit_length_distributions`` honors turn count, lengths, and delay."""
    body = "hello world " * 80
    mock_load_pairs.return_value = [(body, "line\n" * 60)] * 400

    tok = _FakeTokenizer(model_max_length=50_000)
    dataset = InstructCoderBenchmarkDataset()
    dataset.dataset_path = "/tmp/instruct_coder_mock.json"

    with TokenizerPool(tok, loader=_fake_loader) as pool:
        samples = dataset.gen_multiturn_sessions(
            num_sessions=4,
            tokenizer=tok,
            pool=pool,
            shuffle=False,
            fit_length_distributions=True,
            num_turns="DU(3,3)",
            input_len="80",
            output_len="20",
            delay_between_turns_dist="100",
            sys_prompt_ratio=0.0,
        )

    assert len(samples.chat_sessions) == 4
    for session in samples.chat_sessions:
        assert len(session.messages) == 6
        for assistant in session.messages[1::2]:
            assert assistant.source == "assistant"
            assert assistant.num_tokens == 20
            assert assistant.delay_until_next_message == 100.0
        for user in session.messages[0::2]:
            assert user.source == "user"
            assert user.num_tokens == 80


def _write_chat_judge_file(path: Path) -> None:
    """Write a single chat-judge session: system turn + 3 user turns."""
    path.write_text(
        json.dumps(
            {
                "session_id": "s0",
                "turns": [
                    {"text": "You are a judge.", "role": "system"},
                    {"text": "Item 0"},
                    {"text": "Item 1"},
                    {"text": "Item 2"},
                ],
            }
        )
        + "\n"
    )


def test_chat_judge_gen_chat_sessions_sets_per_turn_delay(
    tmp_path: Path,
) -> None:
    """The delay is sampled per turn and set on every user message except
    the final one; the system message never carries a delay."""
    dataset_file = tmp_path / "chat_judge.jsonl"
    _write_chat_judge_file(dataset_file)

    dataset = ChatJudgeBenchmarkDataset()
    dataset.dataset_path = str(dataset_file)

    samples = dataset.gen_chat_sessions(
        num_sessions=1,
        tokenizer=_FakeTokenizer(),
        shuffle=False,
        delay_between_turns_dist="100",
    )

    (session,) = samples.chat_sessions
    assert session.messages[0].source == "system"
    assert session.messages[0].delay_until_next_message is None

    user_messages = [m for m in session.messages if m.source == "user"]
    assert [m.delay_until_next_message for m in user_messages] == [
        100.0,
        100.0,
        None,
    ]


def test_chat_judge_gen_chat_sessions_no_delay_by_default(
    tmp_path: Path,
) -> None:
    """Without a delay distribution, no message carries an inter-turn delay."""
    dataset_file = tmp_path / "chat_judge.jsonl"
    _write_chat_judge_file(dataset_file)

    dataset = ChatJudgeBenchmarkDataset()
    dataset.dataset_path = str(dataset_file)

    samples = dataset.gen_chat_sessions(
        num_sessions=1,
        tokenizer=_FakeTokenizer(),
        shuffle=False,
    )

    (session,) = samples.chat_sessions
    assert all(m.delay_until_next_message is None for m in session.messages)


def test_every_user_body_is_marked_distinct() -> None:
    """Every body carries its own ``[N] `` marker, wrap or no wrap."""
    tok = _FakeTokenizer(model_max_length=50_000)
    pool_texts = ["alpha body text", "beta body text", "gamma body text"]
    num_sessions = 5
    turns_per_session = 2  # 10 total turns, pool only has 3 -> wraps

    with TokenizerPool(tok, loader=_fake_loader) as pool:
        samples = build_chat_samples_from_user_text_pool(
            pool=pool,
            user_text_pool=pool_texts,
            num_sessions=num_sessions,
            num_turns=str(turns_per_session),
            input_len="80",
            output_len="20",
            delay_between_turns_dist=None,
            sys_prompt_ratio=0.0,
            max_num_unique_sys_prompt=1,
            shuffle_pool=False,
            log_prefix="test-wrap",
        )

    assert len(samples.chat_sessions) == num_sessions
    user_contents: list[str] = []
    for session in samples.chat_sessions:
        for user in session.messages[0::2]:
            assert user.source == "user"
            user_contents.append(user.content)

    assert len(user_contents) == num_sessions * turns_per_session

    # Numbered from the first body, monotonically, across the whole run.
    for index, content in enumerate(user_contents):
        assert content.startswith(f"[{index}] "), (
            f"expected '[{index}] ' prefix, got: {content!r}"
        )

    # Marker token budget reservation keeps on-the-wire length close to target.
    target_in = 80
    for user in (
        msg
        for session in samples.chat_sessions
        for msg in session.messages[0::2]
    ):
        assert abs(user.num_tokens - target_in) <= 4


def test_truncated_sessions_logged_once_in_aggregate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Context overflow is reported as one aggregate line, not one log line
    per truncated session."""
    # model_max_length is small enough that a second turn always overflows the
    # 0.95 * max_context budget, so every session is truncated after turn 0.
    tok = _FakeTokenizer(model_max_length=200)
    pool_texts = ["alpha body text", "beta body text", "gamma body text"]
    num_sessions = 5

    module = (
        "max.benchmark.benchmark_shared.datasets.multiturn_distribution_fit"
    )
    with caplog.at_level(logging.INFO, logger=module):
        with TokenizerPool(tok, loader=_fake_loader) as pool:
            samples = build_chat_samples_from_user_text_pool(
                pool=pool,
                user_text_pool=pool_texts,
                num_sessions=num_sessions,
                num_turns="3",  # planned turns > 1, but only turn 0 fits
                input_len="80",
                output_len="20",
                delay_between_turns_dist=None,
                sys_prompt_ratio=0.0,
                max_num_unique_sys_prompt=1,
                shuffle_pool=False,
                log_prefix="test-truncate",
            )

    # Each session kept exactly its first turn (one user + one assistant).
    assert len(samples.chat_sessions) == num_sessions
    for session in samples.chat_sessions:
        assert len(session.messages) == 2

    truncation_logs = [
        record
        for record in caplog.records
        if "truncated to fit" in record.getMessage()
    ]
    assert len(truncation_logs) == 1, (
        "expected a single aggregate truncation log, got: "
        f"{[record.getMessage() for record in truncation_logs]}"
    )
    assert (
        f"{num_sessions}/{num_sessions} sessions truncated"
        in truncation_logs[0].getMessage()
    )


_BASH_TOOL_ANTHROPIC = AnthropicTool(
    id="bash",
    description="run a shell command",
    inputSchema={
        "jsonSchema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
    },
)
_TODOREAD_TOOL_ANTHROPIC = AnthropicTool(
    id="todoread",
    description="",
    inputSchema={"jsonSchema": {}},
)


def _mock_nemotron_rows() -> list[dict[str, object]]:
    """Three synthetic rows mimicking the Nemotron-SFT-OpenCode-v1 shape.

    ``tools`` is in Anthropic schema (``{id, description, inputSchema:
    {jsonSchema}}``) to match the real dataset, which records OpenCode CLI
    sessions whose tool definitions follow the Anthropic API.

    Note that ``content`` is deliberately heterogeneous: a plain ``str`` in
    most messages but a list of content blocks in the first assistant turn.
    This mirrors the upstream corpus and is exactly the shape that broke the
    old pyarrow-backed loader (PERF-2714).
    """
    return [
        {
            "messages": [
                {"role": "system", "content": "agent prompt"},
                {"role": "user", "content": "edit file a.py"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok, calling bash"}],
                },
                {"role": "tool", "content": "stdout"},
                {"role": "user", "content": "next step"},
                {"role": "assistant", "content": "done, here is the diff"},
            ],
            "tools": [msgspec.to_builtins(_BASH_TOOL_ANTHROPIC)],
        },
        {
            # Trace with no usable assistant reply — should be skipped by
            # both sample_requests() and gen_multiturn_sessions().
            "messages": [
                {"role": "system", "content": "agent prompt"},
                {"role": "user", "content": "hello"},
                {"role": "tool", "content": "result"},
            ],
            "tools": [],
        },
        {
            "messages": [
                {"role": "user", "content": "write hello world"},
                {"role": "assistant", "content": "print('hello world')"},
            ],
            "tools": [msgspec.to_builtins(_TODOREAD_TOOL_ANTHROPIC)],
        },
    ]


def _mock_nemotron_jsonl_lines() -> list[bytes]:
    """Encode :func:`_mock_nemotron_rows` as raw JSONL byte lines.

    This matches what :meth:`NemotronOpenCodeBenchmarkDataset._stream_jsonl_lines`
    yields, so patching that method drives the real msgspec line decoder.
    """
    return [json.dumps(row).encode("utf-8") for row in _mock_nemotron_rows()]


@patch(
    "max.benchmark.benchmark_shared.datasets.nemotron_opencode"
    ".NemotronOpenCodeBenchmarkDataset._stream_jsonl_lines"
)
def test_nemotron_opencode_sample_requests(mock_stream: Mock) -> None:
    """``sample_requests`` collapses each row to (history, last_assistant)."""
    mock_stream.return_value = iter(_mock_nemotron_jsonl_lines())

    tok = Mock(spec=PreTrainedTokenizerBase)
    tok.encode = Mock(
        side_effect=lambda text, add_special_tokens=False: (
            [0] * max(len(text), 1)
        )
    )

    dataset = NemotronOpenCodeBenchmarkDataset()
    samples = dataset.sample_requests(
        num_requests=5,
        tokenizer=tok,
        shuffle=False,
        min_prompt_len=1,
        min_output_len=1,
    )

    # Row 1 (multi-turn ending on assistant) and row 3 (single user/assistant)
    # both produce samples; row 2 (no usable assistant reply) is filtered out.
    assert len(samples.requests) == 2
    assert all(isinstance(r, SampledRequest) for r in samples.requests)
    assert all(isinstance(r.prompt_formatted, list) for r in samples.requests)
    # ``last_loaded_tool_schemas`` retains the raw Anthropic schemas straight
    # from the dataset rows (for inspection / follow-up work).
    assert dataset.last_loaded_tool_schemas == [
        [_BASH_TOOL_ANTHROPIC],
        [_TODOREAD_TOOL_ANTHROPIC],
    ]
    # ``SampledRequest.tools`` carries the OpenAI-translated schemas so the
    # chat-completions driver can forward them as ``tools=[...]`` against the
    # MAX OpenAI-compatible server.
    assert [r.tools for r in samples.requests] == [
        [_anthropic_tool_to_openai(_BASH_TOOL_ANTHROPIC)],
        [_anthropic_tool_to_openai(_TODOREAD_TOOL_ANTHROPIC)],
    ]
    # ``ignore_eos=True`` always, matching sharegpt/arxiv (decode the full
    # target length even when ``output_len`` came from the dataset).
    assert all(r.ignore_eos is True for r in samples.requests)
    # The raw JSONL stream was consumed exactly once.
    mock_stream.assert_called_once()


@patch(
    "max.benchmark.benchmark_shared.datasets.nemotron_opencode"
    ".NemotronOpenCodeBenchmarkDataset._stream_jsonl_lines"
)
def test_nemotron_opencode_disable_tool_calls(mock_stream: Mock) -> None:
    """``enable_tool_calls=False`` drops rows containing tool messages."""
    mock_stream.return_value = iter(_mock_nemotron_jsonl_lines())

    tok = Mock(spec=PreTrainedTokenizerBase)
    tok.encode = Mock(
        side_effect=lambda text, add_special_tokens=False: (
            [0] * max(len(text), 1)
        )
    )

    dataset = NemotronOpenCodeBenchmarkDataset()
    samples = dataset.sample_requests(
        num_requests=5,
        tokenizer=tok,
        shuffle=False,
        enable_tool_calls=False,
        min_prompt_len=1,
        min_output_len=1,
    )

    # Only row 3 has zero tool/non-chat messages.
    assert len(samples.requests) == 1
    # With tool calls disabled, ``tools`` is cleared even when the row has
    # schemas attached.
    assert samples.requests[0].tools is None


@patch(
    "max.benchmark.benchmark_shared.datasets.nemotron_opencode"
    ".NemotronOpenCodeBenchmarkDataset._stream_jsonl_lines"
)
def test_nemotron_opencode_gen_multiturn(mock_stream: Mock) -> None:
    """Multi-turn conversion alternates user/assistant and drops tools."""
    mock_stream.return_value = iter(_mock_nemotron_jsonl_lines())

    tok = Mock(spec=PreTrainedTokenizerBase)
    tok.encode = Mock(
        side_effect=lambda text, add_special_tokens=False: (
            [0] * max(len(text), 1)
        )
    )

    dataset = NemotronOpenCodeBenchmarkDataset()
    samples = dataset.gen_multiturn_sessions(
        num_sessions=5,
        tokenizer=tok,
        shuffle=False,
        min_input_len=1,
        min_output_len=1,
    )

    # Row 1 yields 2 user/assistant turns, row 2 has no assistant pair and
    # is dropped, and row 3 yields 1 turn.
    assert len(samples.chat_sessions) == 2
    for session in samples.chat_sessions:
        # Alternation must hold.
        for i, msg in enumerate(session.messages):
            assert msg.source == ("user" if i % 2 == 0 else "assistant")
        # Assistant content placeholders are empty (filled by the live model
        # at run time, like agentic-code / instruct-coder).
        for assistant in session.messages[1::2]:
            assert assistant.content == ""


@patch(
    "max.benchmark.benchmark_shared.datasets.nemotron_opencode"
    ".NemotronOpenCodeBenchmarkDataset._stream_jsonl_lines"
)
def test_nemotron_opencode_iter_rows_heterogeneous_content(
    mock_stream: Mock,
) -> None:
    """Regression for PERF-2714: ``content`` may be a string or a list.

    The old ``datasets.load_dataset`` path unified a single Arrow schema
    across rows via pyarrow, which cannot represent ``messages[].content``
    being a ``str`` in one row and a list of content blocks in another. It
    aborted with ``ArrowInvalid: ... changed from string to array`` and then a
    whole-file ``pandas.read_json`` fallback failed with
    ``ValueError: Trailing data``. Decoding each JSONL line independently with
    msgspec must handle both shapes without error.
    """
    lines = [
        json.dumps(
            {"messages": [{"role": "user", "content": "plain string"}]}
        ).encode("utf-8"),
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "block list"}],
                    }
                ]
            }
        ).encode("utf-8"),
        # A malformed line must be skipped, not abort the whole stream.
        b"{ this is not valid json",
        # An empty line is also tolerated.
        b"",
    ]
    mock_stream.return_value = iter(lines)

    dataset = NemotronOpenCodeBenchmarkDataset()
    rows = list(dataset._iter_rows())

    # The empty line is dropped by ``_stream_jsonl_lines`` in production; here
    # the mock yields it directly, so msgspec's decoder rejects it. Either
    # way, only the two well-formed rows survive.
    assert len(rows) == 2
    assert rows[0].messages is not None
    assert rows[0].messages[0].content == "plain string"
    assert rows[1].messages is not None
    content = rows[1].messages[0].content
    assert isinstance(content, list)
    assert content[0].text == "block list"


def test_nemotron_opencode_stream_path_uses_repo_id() -> None:
    """``_stream_jsonl_lines`` targets the dataset blob under ``datasets/``."""
    dataset = NemotronOpenCodeBenchmarkDataset()
    dataset.subset = "general"
    captured: dict[str, str] = {}

    class _FakeFile:
        def __enter__(self) -> _FakeFile:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def __iter__(self):
            return iter([b'{"messages": []}\n'])

    class _FakeFs:
        def open(self, path: str, mode: str) -> _FakeFile:
            captured["path"] = path
            captured["mode"] = mode
            return _FakeFile()

    with patch(
        "max.benchmark.benchmark_shared.datasets.nemotron_opencode"
        ".HfFileSystem",
        return_value=_FakeFs(),
    ):
        lines = list(dataset._stream_jsonl_lines())

    assert lines == [b'{"messages": []}']
    assert captured["mode"] == "rb"
    assert (
        captured["path"]
        == f"datasets/{NEMOTRON_OPENCODE_REPO_ID}/general/data.jsonl"
    )


def test_nemotron_opencode_rejects_unknown_subset() -> None:
    dataset = NemotronOpenCodeBenchmarkDataset()
    dataset.subset = "does-not-exist"
    tok = Mock(spec=PreTrainedTokenizerBase)
    with pytest.raises(ValueError, match="Unknown Nemotron-OpenCode subset"):
        dataset.sample_requests(num_requests=1, tokenizer=tok)


def test_nemotron_opencode_rejects_dataset_path() -> None:
    """``--dataset-path`` is not supported; surface that explicitly."""
    dataset = NemotronOpenCodeBenchmarkDataset()
    dataset.dataset_path = "/tmp/some-local.jsonl"
    with pytest.raises(
        ValueError, match="nemotron-opencode does not support --dataset-path"
    ):
        dataset.fetch()


def test_anthropic_tool_to_openai_canonical() -> None:
    """Canonical Anthropic shape maps to OpenAI ``{type, function}``."""
    out = _anthropic_tool_to_openai(_BASH_TOOL_ANTHROPIC)
    assert out == {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }


def test_anthropic_tool_to_openai_inputschema_without_jsonschema_wrapper() -> (
    None
):
    """Some rows put the JSON Schema directly under ``inputSchema``."""
    tool = AnthropicTool(
        id="ls",
        description="list files",
        inputSchema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
    )
    out = _anthropic_tool_to_openai(tool)
    assert out["function"]["parameters"] == {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }


def test_anthropic_tool_to_openai_missing_fields() -> None:
    """Missing ``inputSchema`` / ``description`` collapse to safe defaults."""
    assert _anthropic_tool_to_openai(AnthropicTool(id="noop")) == {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "",
            "parameters": {},
        },
    }


# ===----------------------------------------------------------------------=== #
# Artificial Analysis corpus generator
# ===----------------------------------------------------------------------=== #


def _prompt_text(req: SampledRequest) -> str:
    """Narrow an AA text request's ``prompt_formatted`` to ``str``.

    AA requests always carry a plain string prompt (no chat messages), so this
    asserts the runtime type and satisfies MyPy for the ``str | list`` union.
    """
    assert isinstance(req.prompt_formatted, str)
    return req.prompt_formatted


def _make_aa_dataset(articles: list[str]) -> ArtificialAnalysisBenchmarkDataset:
    """AA dataset with ``_iter_articles`` stubbed to a fixed article list."""
    dataset = ArtificialAnalysisBenchmarkDataset()
    dataset._iter_articles = lambda *a, **k: iter(articles)  # type: ignore[method-assign]
    return dataset


def test_aa_in_registry() -> None:
    """The artificial-analysis dataset resolves through the registry."""
    assert "artificial-analysis" in DATASET_REGISTRY
    assert (
        DATASET_REGISTRY["artificial-analysis"].class_name
        == "ArtificialAnalysisBenchmarkDataset"
    )


def test_aa_requires_input_len() -> None:
    """input_len is mandatory."""
    dataset = _make_aa_dataset(["word " * 100])
    with pytest.raises(ValueError, match="input_len is required"):
        dataset.sample_requests(num_requests=1, tokenizer=_FakeTokenizer())


def test_aa_sizes_prompts_to_budget_and_rotates_tasks() -> None:
    """Prompts stay within the model-tokenizer budget and rotate tasks."""
    # Plenty of long articles so each prompt can be filled to budget. The fake
    # tokenizer counts ~1 token/char, so input_len must exceed the template
    # overhead (a couple hundred chars).
    articles = [f"alpha{i} " * 500 for i in range(20)]
    dataset = _make_aa_dataset(articles)
    input_len = 2000

    samples = dataset.sample_requests(
        num_requests=4,
        tokenizer=_FakeTokenizer(),
        output_lengths=[1000, 1500, 2000, 1000],
        input_len=input_len,
    )

    requests = samples.requests
    assert len(requests) == 4

    # Balanced over AA's full task mix: distinct task prefixes per request.
    leading_lines = {_prompt_text(r).split("\n", 1)[0] for r in requests}
    assert len(leading_lines) == 4

    for r, expected_out in zip(requests, [1000, 1500, 2000, 1000], strict=True):
        # Whole prompt is sized to the model-tokenizer budget.
        assert 0 < r.prompt_len <= input_len
        # Output budget is honored and EOS is ignored to force min length.
        assert r.output_len == expected_out
        assert r.ignore_eos is True
        assert r.encoded_images == []


def test_aa_without_output_lengths_leaves_eos_enabled() -> None:
    """With no output budget, requests don't force generation length."""
    articles = [f"beta{i} " * 500 for i in range(8)]
    dataset = _make_aa_dataset(articles)

    samples = dataset.sample_requests(
        num_requests=2,
        tokenizer=_FakeTokenizer(),
        input_len=2000,
    )

    for r in samples.requests:
        assert r.output_len is None
        assert r.ignore_eos is False


def test_aa_sampled_output_len_caps_with_eos_enabled() -> None:
    """A sampled output_len caps max tokens but leaves EOS enabled."""
    articles = [f"gamma{i} " * 500 for i in range(8)]
    dataset = _make_aa_dataset(articles)

    samples = dataset.sample_requests(
        num_requests=2,
        tokenizer=_FakeTokenizer(),
        input_len=2000,
        output_len=50,  # constant distribution -> deterministic cap
    )

    for r in samples.requests:
        # Output length is set (a runaway cap), but EOS is not ignored.
        assert r.output_len == 50
        assert r.ignore_eos is False


def test_hf_hub_download_retries_on_racy_cache_entry() -> None:
    """A racy `.incomplete` FileNotFoundError triggers one force_download retry."""
    with patch(
        "max.benchmark.benchmark_shared.datasets._hf_download"
        ".huggingface_hub.hf_hub_download",
        side_effect=[
            FileNotFoundError("dangling .incomplete"),
            "/cache/d.json",
        ],
    ) as mock_download:
        assert hf_hub_download_with_retry(repo_id="org/d") == "/cache/d.json"
    assert mock_download.call_args_list[1].kwargs["force_download"] is True


def test_hf_hub_download_does_not_retry_offline_miss() -> None:
    """An offline/uncached miss (LocalEntryNotFoundError) is surfaced at once."""
    with patch(
        "max.benchmark.benchmark_shared.datasets._hf_download"
        ".huggingface_hub.hf_hub_download",
        side_effect=hf_hub_errors.LocalEntryNotFoundError("offline"),
    ) as mock_download:
        with pytest.raises(hf_hub_errors.LocalEntryNotFoundError):
            hf_hub_download_with_retry(repo_id="org/d")
    assert mock_download.call_count == 1


def _has_system_block(content: str) -> bool:
    # build_scaled_user_message joins [system, body] with a blank line, and
    # emits a single part when there is no system block.
    return "\n\n" in content


def test_system_prefix_lands_on_the_first_turn_only() -> None:
    """A session resends its history, so a later system block is a duplicate."""
    tok = _FakeTokenizer(model_max_length=50_000)

    with TokenizerPool(tok, loader=_fake_loader) as pool:
        samples = build_chat_samples_from_user_text_pool(
            pool=pool,
            user_text_pool=[f"body-{i} " * 40 for i in range(20)],
            num_sessions=3,
            num_turns="4",
            input_len="400",
            output_len="20",
            delay_between_turns_dist=None,
            sys_prompt_ratio=0.5,
            max_num_unique_sys_prompt=1,
            shuffle_pool=False,
            log_prefix="test-sys",
        )

    assert samples.chat_sessions
    for session in samples.chat_sessions:
        users = [m for m in session.messages if m.source == "user"]
        assert len(users) == 4
        assert _has_system_block(users[0].content)
        for later in users[1:]:
            assert not _has_system_block(later.content)

    # The prefix must still reach warmup now that only turn 0 carries one.
    assert samples.shared_contexts
