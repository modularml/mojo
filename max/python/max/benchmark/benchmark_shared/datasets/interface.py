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

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from ._tokenizer_pool import TokenizerPool
from .registry import DATASET_REGISTRY
from .types import RequestSamples


class BenchmarkDataset(ABC):
    """Abstract base class for benchmark datasets.

    This class provides a common interface for working with different types of
    benchmark datasets, whether they are fetched from HuggingFace Hub or loaded
    from local files. It handles automatic dataset fetching, validation, and
    provides a standardized interface for sampling requests.

    Usage:
        Subclasses must implement fetch() to specify how to download
        and sample their specific datasets. Subclasses must also implement sample_requests()
        to specify how to sample requests from the dataset.

        Example initialization patterns:

        # Auto-fetch from HuggingFace Hub
        dataset = BenchmarkDataset.from_flags(dataset_name="sharegpt")

        # Use local file
        dataset = BenchmarkDataset.from_flags(dataset_path="/path/to/local/dataset.json")

        # Sample requests
        requests = dataset.sample_requests(
            num_requests=100,
            tokenizer=tokenizer,
            input_len=1024,
            output_len=512
        )

    Subclass Requirements:
        - Must implement fetch() -> None to handle both local and HuggingFace dataset loading
        - Must implement sample_requests(**kwargs) -> list[SampledRequest]
        - Should raise ValueError for unsupported dataset names or invalid configurations
        - May raise NotImplementedError if the requested mode is not supported
        - Should handle only dataset types relevant to their domain

    Raises:
        ValueError: If neither dataset_name nor dataset_path is provided during initialization
    """

    dataset_name: str | None = None
    dataset_path: str | None = None
    has_multiturn_chat_support: bool = False

    @classmethod
    def from_flags(
        cls,
        dataset_name: str | None = None,
        dataset_path: str | None = None,
    ) -> BenchmarkDataset:
        """Factory method to create the appropriate dataset subclass based on dataset name.

        This factory method automatically selects and instantiates the correct
        BenchmarkDataset subclass based on the provided dataset_name. This eliminates
        the need for callers to know which specific subclass to instantiate.

        Args:
            dataset_name: Name of the dataset. Used to determine
                which subclass to instantiate. If None, dataset_path must be provided.
            dataset_path: Local path to the dataset file. If provided,
                takes precedence over automatic fetching.

        Returns:
            BenchmarkDataset: An instance of the appropriate subclass

        Raises:
            ValueError: If dataset_name is not recognized or if both dataset_name
                and dataset_path are None

        Example:
            # Creates ShareGPTBenchmarkDataset instance from HuggingFace
            dataset = BenchmarkDataset.from_flags(dataset_name="sharegpt")

            # Creates appropriate subclass with local file
            dataset = BenchmarkDataset.from_flags(
                dataset_name="sharegpt",
                dataset_path="/local/file.json",
            )

            # Creates dataset using environment variable for local path
            dataset = BenchmarkDataset.from_flags(
                dataset_name="sharegpt",
            )
        """
        if not dataset_name and not dataset_path:
            raise ValueError(
                "Either dataset_name or dataset_path must be provided"
            )
        elif dataset_path is not None and not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist")
        # If we have a dataset_path but no dataset_name, we can't determine the subclass
        if not dataset_name:
            raise ValueError(
                "dataset_name is required to determine the appropriate dataset"
                " subclass. Cannot infer subclass from dataset_path alone."
            )

        # Get the dataset class based on dataset_name
        dataset_class = cls._get_dataset_class(dataset_name)

        instance = dataset_class()
        instance.dataset_name = dataset_name
        instance.dataset_path = dataset_path
        instance.has_multiturn_chat_support = DATASET_REGISTRY[
            dataset_name
        ].has_multiturn_chat_support

        instance.fetch()

        return instance

    @classmethod
    def _get_dataset_class(cls, dataset_name: str) -> type[BenchmarkDataset]:
        """Get the appropriate dataset class for the given dataset name.

        Args:
            dataset_name: Name of the dataset

        Returns:
            The appropriate BenchmarkDataset subclass

        Raises:
            ValueError: If dataset_name is not recognized
        """

        if dataset_name not in DATASET_REGISTRY:
            available_datasets = ", ".join(DATASET_REGISTRY.keys())
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available datasets: {available_datasets}"
            )

        # Dynamically resolve the class at runtime
        dataset_entry = DATASET_REGISTRY[dataset_name]
        class_name = dataset_entry.class_name
        from .. import datasets

        dataset_class = getattr(datasets, class_name)

        if dataset_class is None:
            raise ValueError(f"Dataset class {class_name} not found")

        return dataset_class

    def __str__(self) -> str:
        """Return a user-friendly string representation of the dataset.

        Returns:
            String representation showing dataset path (if local) or name and class type
        """
        class_name = self.__class__.__name__
        if self.dataset_path:
            return f"local_dataset_at_{self.dataset_path} ({class_name})"
        elif self.dataset_name:
            return f"{self.dataset_name} ({class_name})"
        else:
            return f"uninitialized_dataset ({class_name})"

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging.

        Returns:
            Detailed string representation with all attributes
        """
        return (
            f"{self.__class__.__name__}("
            f"dataset_name='{self.dataset_name}', "
            f"dataset_path='{self.dataset_path}', "
            f"has_multiturn_chat_support={self.has_multiturn_chat_support})"
        )

    @abstractmethod
    def fetch(self) -> None:
        """Fetch dataset based on the current dataset_mode and set dataset_path.

        This method handles both local and HuggingFace dataset loading:
        - For LOCAL mode: Uses dataset_path (from constructor or environment variable)
        - For HUGGINGFACE mode: Downloads dataset from HuggingFace Hub

        Raises:
            ValueError: If the dataset is unknown, not supported, or if both modes are specified
            NotImplementedError: If the dataset doesn't support the requested mode
        """

    @abstractmethod
    def sample_requests(
        self,
        num_requests: int,
        tokenizer: PreTrainedTokenizerBase,
        output_lengths: Sequence[int] | None = None,
        shuffle: bool = True,
        *,
        pool: TokenizerPool | None = None,
        **kwargs,
    ) -> RequestSamples:
        """Sample requests from the dataset.

        This is the standardized interface that all dataset implementations must follow.
        Additional dataset-specific parameters can be passed via **kwargs.

        Args:
            num_requests: Number of requests to sample
            tokenizer: Tokenizer for computing token lengths
            output_lengths: Optional sequence of output lengths for each request.
                If None, uses the actual completion lengths from the dataset.
                If provided, must have length equal to num_requests.
            shuffle: Whether to shuffle the dataset before sampling. Default is True.
            pool: Optional tokenizer process pool. Required by datasets that
                fan tokenize work out to workers (e.g. Random/Synthetic).
                Datasets that don't tokenize via the pool ignore it.
            **kwargs: Additional dataset-specific parameters

        Returns:
            Sequence of SampledRequest objects

        Raises:
            ValueError: If the dataset cannot be loaded or parameters are invalid
            NotImplementedError: If required parameters are missing for this dataset type
        """
