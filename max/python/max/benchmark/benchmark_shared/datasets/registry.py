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

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass
class DatasetRegistryEntry:
    """Registry entry for a benchmark dataset."""

    class_name: str
    """The name of the BenchmarkDataset subclass that implements this dataset."""
    has_multiturn_chat_support: bool
    """Whether this dataset supports multiturn chat scenarios."""


"""Registry mapping dataset names to their implementation metadata and capabilities.

This registry serves as the central configuration for all supported benchmark datasets
in the serving benchmarking framework. It enables dynamic class resolution and
capability discovery without requiring explicit imports or hardcoded conditionals.

Structure:
    Each registry entry maps a dataset name (str) to a DatasetRegistryEntry containing:

    - class_name (str): The name of the BenchmarkDataset subclass that implements
      this dataset. The class must be defined in the global namespace of this module.

    - has_multiturn_chat_support (bool): Whether this dataset can generate or handle
      multiturn conversational scenarios. This affects which benchmarking modes
      and evaluation patterns can be applied to the dataset.

Adding New Datasets:
    To register a new dataset:

    1. Implement a new BenchmarkDataset subclass with required methods:
       - fetch() for both local and HuggingFace Hub integration
       - sample_requests() for generating benchmark requests

    2. Add an entry to this registry:
       "my_dataset": DatasetRegistryEntry(
           class_name="MyDatasetBenchmarkDataset",
           has_multiturn_chat_support=True,  # or False
       )

    3. Document the dataset in max/python/docs/cli/benchmark.rst (Datasets
       section): name, source, required flags, and any dataset-specific CLI
       options.

Notes:
    - The registry is loaded at module import time and should not be modified
      at runtime unless you understand the implications for ongoing benchmarks
    - Multiturn support affects memory usage and complexity of benchmark scenarios
    - Some datasets may support multiturn technically but set it to False due
      to domain-specific constraints or intended usage patterns
"""

DATASET_REGISTRY: Mapping[str, DatasetRegistryEntry] = {
    "artificial-analysis": DatasetRegistryEntry(
        class_name="ArtificialAnalysisBenchmarkDataset",
        has_multiturn_chat_support=False,
    ),
    "arxiv-summarization": DatasetRegistryEntry(
        class_name="ArxivSummarizationBenchmarkDataset",
        has_multiturn_chat_support=False,
    ),
    "axolotl": DatasetRegistryEntry(
        class_name="AxolotlBenchmarkDataset",
        has_multiturn_chat_support=False,
    ),
    "batch-job": DatasetRegistryEntry(
        class_name="BatchJobBenchmarkDataset",
        has_multiturn_chat_support=False,
    ),
    "code_debug": DatasetRegistryEntry(
        class_name="CodeDebugBenchmarkDataset",
        has_multiturn_chat_support=True,
    ),
    "chat-judge": DatasetRegistryEntry(
        class_name="ChatJudgeBenchmarkDataset",
        has_multiturn_chat_support=True,
    ),
    "obfuscated-conversations": DatasetRegistryEntry(
        class_name="ObfuscatedConversationsBenchmarkDataset",
        has_multiturn_chat_support=False,
    ),
    "local-image": DatasetRegistryEntry(
        class_name="LocalImageBenchmarkDataset",
        has_multiturn_chat_support=False,
    ),
    "random": DatasetRegistryEntry(
        class_name="RandomBenchmarkDataset",
        has_multiturn_chat_support=True,
    ),
    "synthetic": DatasetRegistryEntry(
        class_name="SyntheticBenchmarkDataset",
        has_multiturn_chat_support=True,
    ),
    "sharegpt": DatasetRegistryEntry(
        class_name="ShareGPTBenchmarkDataset",
        has_multiturn_chat_support=False,
    ),
    "sonnet": DatasetRegistryEntry(
        class_name="SonnetBenchmarkDataset",
        has_multiturn_chat_support=False,
    ),
    "vision-arena": DatasetRegistryEntry(
        class_name="VisionArenaBenchmarkDataset",
        has_multiturn_chat_support=False,
    ),
    "synthetic-pixel": DatasetRegistryEntry(
        class_name="SyntheticPixelBenchmarkDataset",
        has_multiturn_chat_support=False,
    ),
    "agentic-code": DatasetRegistryEntry(
        class_name="AgenticCodeBenchmarkDataset",
        has_multiturn_chat_support=True,
    ),
    "instruct-coder": DatasetRegistryEntry(
        class_name="InstructCoderBenchmarkDataset",
        has_multiturn_chat_support=True,
    ),
    "nemotron-opencode": DatasetRegistryEntry(
        class_name="NemotronOpenCodeBenchmarkDataset",
        has_multiturn_chat_support=True,
    ),
}
