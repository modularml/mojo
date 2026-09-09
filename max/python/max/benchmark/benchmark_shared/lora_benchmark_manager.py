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

"""LoRA adapter generation and management utilities for benchmarking."""

import asyncio
import json
import logging
import os
import random

# The import below breaks CI due to some version driver issue
# from max.pipelines.lib.lora import ADAPTER_CONFIG_FILE
ADAPTER_CONFIG_FILE = "adapter_config.json"
from tqdm.asyncio import tqdm

from .metrics import LoRAMetrics
from .request import async_request_lora_load, async_request_lora_unload

logger = logging.getLogger(__name__)


# TODO: make lora_configs a dataclass type than a dict[str, str]
class LoRABenchmarkManager:
    """Manages LoRA adapters for benchmarking including configuration,
    loading/unloading, and traffic distribution.

    This class coordinates all LoRA-related operations for benchmarking:
    - Parses LoRA adapter paths and configurations
    - Reads adapter metadata (rank, target modules, etc.)
    - Optionally manages traffic distribution across adapters
    - Provides interface for loading/unloading operations
    """

    def __init__(
        self,
        lora_paths: list[str],
        num_requests: int = 0,
        traffic_ratios: list[float] | None = None,
        uniform_ratio: float | None = None,
        max_num_loras: int | None = None,
        seed: int | None = None,
        max_concurrent_lora_ops: int = 1,
    ):
        """Initialize LoRA benchmark manager.

        Args:
            lora_paths: List of paths to LoRA adapters (format: "path" or "name=path")
            num_requests: Total number of requests (required if using traffic distribution)
            traffic_ratios: List of traffic ratios for each LoRA adapter.
                           If provided, must have same length as lora_paths.
                           Sum must not exceed 1.0. Remainder goes to base model.
            uniform_ratio: If traffic_ratios not provided, probability of
                          selecting any LoRA uniformly at random (vs base model)
            seed: Random seed for shuffling (optional)
        """
        self.lora_paths = lora_paths
        self.max_concurrent_lora_ops = max_concurrent_lora_ops
        self._parse_configs()

        self.traffic_selector = LoRATrafficSelector(
            self.lora_configs,
            num_requests,
            traffic_ratios,
            uniform_ratio,
            seed,
        )

        # Initialize metrics tracking
        self.metrics = LoRAMetrics()

    def _parse_configs(self) -> None:
        """Parse LoRA paths and read adapter configurations."""
        max_rank: int = -1
        lora_configs: dict[str, str] = {}

        for i, path in enumerate(self.lora_paths):
            if "=" in path:
                name, path = path.split("=")
            else:
                name = f"adapter_{i}"
            abs_path = os.path.abspath(path)
            lora_configs[name] = f"{abs_path}"

            config_path = os.path.join(path, ADAPTER_CONFIG_FILE)
            if not os.path.exists(config_path):
                raise ValueError(
                    f"Adapter config file not found: {config_path}"
                )

            with open(config_path) as f:
                adapter_config = json.load(f)
            max_rank = max(adapter_config["r"], max_rank)

        self.lora_configs = lora_configs
        self.max_rank = max_rank

    def get_lora_for_request(self, request_idx: int) -> str | None:
        """Get LoRA name for a given request index.

        Args:
            request_idx: The request index

        Returns:
            LoRA name or None (for base model)

        Raises:
            ValueError: If traffic selector is not configured
        """
        return self.traffic_selector.get_lora_id(request_idx)

    def log_traffic_distribution(self) -> None:
        """Get string describing the traffic distribution.

        Returns:
            Human-readable description of traffic distribution,
            or "No traffic selector configured" if not set up.
        """
        logger.info(self.traffic_selector)

    async def benchmark_loading(
        self,
        api_url: str,
    ) -> None:
        """Benchmark LoRA loading performance.

        Args:
            api_url: Base API URL
            max_concurrent: Maximum concurrent loading operations
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_lora_ops)

        async def load_with_semaphore(name: str, path: str) -> None:
            async with semaphore:
                success, load_time = await async_request_lora_load(
                    api_url, name, path
                )
                if success:
                    self.metrics.load_times_ms.append(load_time)
                    self.metrics.total_loads += 1
                else:
                    logger.warning(f"Failed to load LoRA '{name}'")

        tasks = [
            load_with_semaphore(name, path)
            for name, path in self.lora_configs.items()
        ]
        await tqdm.gather(*tasks, desc="Loading LoRAs...")

    async def benchmark_unloading(
        self,
        api_url: str,
    ) -> None:
        """Benchmark LoRA unloading performance.

        Args:
            api_url: Base API URL
            max_concurrent: Maximum concurrent unloading operations
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_lora_ops)

        async def unload_with_semaphore(name: str) -> None:
            async with semaphore:
                success, unload_time = await async_request_lora_unload(
                    api_url, name
                )
                if success:
                    self.metrics.unload_times_ms.append(unload_time)
                    self.metrics.total_unloads += 1
                else:
                    logger.warning(f"Failed to unload LoRA '{name}'")

        tasks = [unload_with_semaphore(name) for name in self.lora_configs]
        await tqdm.gather(*tasks, desc="Unloading LoRAs...")


class LoRATrafficSelector:
    """Handles LoRA adapter selection based on traffic patterns."""

    def __init__(
        self,
        lora_configs: dict[str, str],
        num_requests: int,
        traffic_ratios: list[float] | None = None,
        uniform_ratio: float | None = None,
        seed: int | None = None,
    ):
        """Initialize LoRA traffic selector.

        Args:
            lora_configs: Dictionary mapping LoRA names to their paths
            num_requests: Total number of requests in the benchmark
            traffic_ratios: List of traffic ratios for each LoRA adapter.
                           If provided, must have same length as lora_configs.
                           Sum must not exceed 1.0. Remainder goes to base model.
            uniform_ratio: If traffic_ratios not provided, probability of
                          selecting any LoRA uniformly at random (vs base model)
            seed: Random seed for shuffling (optional)
        """
        self.lora_configs = lora_configs
        self.lora_names = list(lora_configs.keys())
        self.num_requests = num_requests
        self.traffic_ratios = traffic_ratios
        self.uniform_ratio = uniform_ratio or 0.0

        if traffic_ratios:
            if len(traffic_ratios) != len(self.lora_names):
                raise ValueError(
                    f"Number of traffic ratios ({len(traffic_ratios)}) "
                    f"must match number of LoRA adapters ({len(self.lora_names)})"
                )

            total_traffic = sum(traffic_ratios)
            if total_traffic > 1.0:
                raise ValueError(
                    f"Sum of traffic ratios ({total_traffic}) cannot exceed 1.0"
                )

        self._request_mapping = self._build_mapping(seed)

    def _build_mapping(self, seed: int | None) -> list[str | None]:
        """Build mapping from request index to LoRA name (or None for base)."""
        if not self.traffic_ratios:
            return []

        # Use traffic ratios to build deterministic mapping
        base_traffic = 1.0 - sum(self.traffic_ratios)
        traffic_ratios = [base_traffic] + self.traffic_ratios

        # Convert ratios to request counts
        request_counts = [
            int(self.num_requests * ratio) for ratio in traffic_ratios
        ]

        # Distribute remaining requests due to rounding
        remaining = self.num_requests - sum(request_counts)
        for i in range(remaining):
            request_counts[i % len(request_counts)] += 1

        # Create shuffled indices
        rng = random.Random(seed) if seed is not None else random
        all_indices = list(range(self.num_requests))
        rng.shuffle(all_indices)

        # Build mapping list
        mapping: list[str | None] = [None] * self.num_requests
        idx = 0

        # Base model requests (None)
        for _ in range(request_counts[0]):
            mapping[all_indices[idx]] = None
            idx += 1

        # LoRA adapter requests
        for lora_idx, lora_name in enumerate(self.lora_names):
            count = request_counts[lora_idx + 1]
            for _ in range(count):
                mapping[all_indices[idx]] = lora_name
                idx += 1

        return mapping

    def get_lora_id(self, request_idx: int) -> str | None:
        """Get LoRA ID for a given request index.

        Args:
            request_idx: The request index

        Returns:
            LoRA name or None (for base model)
        """
        if self.traffic_ratios:
            return self._request_mapping[request_idx]
        else:
            if random.random() < self.uniform_ratio:
                return random.choice(self.lora_names)
            return None

    def __repr__(self) -> str:
        """Get string describing the traffic distribution."""
        if self.traffic_ratios:
            base_count = sum(1 for v in self._request_mapping if v is None)
            lora_counts = {
                name: sum(1 for v in self._request_mapping if v == name)
                for name in self.lora_names
            }
            parts = [f"Base={base_count}"] + [
                f"{name}={count}" for name, count in lora_counts.items()
            ]
            return ", ".join(parts)
        else:
            return f"Uniform random (ratio={self.uniform_ratio})"
