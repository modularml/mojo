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
"""In-process generation of pre-tokenized benchmark workload files.

Samples requests from the shared dataset loaders (sharegpt, sonnet,
random, arxiv-summarization, …), tokenizes their prompts, and writes the
workload JSON the offline benchmark harness consumes:

.. code-block:: json

    {"requests": [{"prompt_tokens": [128000, 9906, ...], "max_new_tokens": 128}]}

Both the standalone ``generate_workload.py`` CLI and the offline benchmark
engine's mach/model-worker runners build the file from here, so a
dataset-driven cell samples the same bytes the hand-run tool would, with no
subprocess hop between them.
"""

from __future__ import annotations

import functools
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from max.benchmark.benchmark_shared.datasets import (
    BenchmarkDataset,
    SampledRequest,
)
from max.benchmark.benchmark_shared.datasets._tokenizer_pool import (
    TokenizerPool,
    _default_loader,
)
from max.benchmark.benchmark_shared.utils import get_tokenizer, resolve_revision
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class WorkloadEntry(TypedDict):
    """One request in a workload file: prompt token ids and its output budget."""

    prompt_tokens: list[int]
    max_new_tokens: int


def worker_tokenizer_loader(
    name_or_path: str,
    model_max_length: int | None,
    _pool_trust_remote_code: bool,
    revision: str | None,
    *,
    trust_remote_code: bool,
) -> PreTrainedTokenizerBase:
    """Load a worker's tokenizer honouring this command's ``trust_remote_code``.

    :class:`TokenizerPool` hands its workers a hardcoded ``True`` for the flag,
    so a repo carrying its own tokenizer implementation would have it executed
    in the workers even when the caller declined it — the parent would load the
    built-in implementation and the workers the repository's, which can
    tokenize differently. The pool's value is ignored in favour of the bound
    one; everything else defers to the pool's own loader.

    Module-level (and bound with :func:`functools.partial`) because the pool
    spawns its workers, so the loader is pickled by reference — a closure would
    not survive the trip.
    """
    return _default_loader(
        name_or_path, model_max_length, trust_remote_code, revision
    )


def build_dataset_kwargs(
    dataset: str,
    *,
    random_input_len: str = "512",
    random_output_len: str = "128",
    sonnet_input_len: int = 512,
    sonnet_prefix_len: int = 128,
) -> dict[str, Any]:
    """Build the dataset-specific ``sample_requests`` kwargs for ``dataset``.

    Values are heterogeneous (str, int, float, bool) and forwarded straight
    into :meth:`BenchmarkDataset.sample_requests`'s own untyped ``**kwargs``,
    so the value type is ``Any`` rather than ``object``: the latter would not
    bind to that method's typed parameters under mypy.
    """
    kwargs: dict[str, Any] = {}
    # ``synthetic`` is a RandomBenchmarkDataset subclass and needs the same
    # arguments; keying on "random" alone left it failing on a missing
    # ``input_len``.
    if dataset in ("random", "synthetic"):
        kwargs["input_len"] = random_input_len
        kwargs["output_len"] = random_output_len
        kwargs["sys_prompt_ratio"] = 0.0
        kwargs["max_num_unique_sys_prompt"] = 1
    elif dataset == "sonnet":
        kwargs["input_len"] = sonnet_input_len
        kwargs["prefix_len"] = sonnet_prefix_len
        # Sonnet requires this explicitly and its length math is derived from
        # the templated prompt (``base_prompt_offset``), so the untemplated
        # form would undershoot ``input_len``. The serving benchmark for these
        # workloads posts to /v1/chat/completions, so templating also matches
        # what the online path measures.
        kwargs["apply_chat_template"] = True
    return kwargs


def tokenize_requests(
    requests: list[SampledRequest],
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens_override: int | None,
) -> list[WorkloadEntry]:
    """Convert sampled requests to workload entries with token ids."""
    entries: list[WorkloadEntry] = []
    for req in requests:
        # Tokenize the prompt text to get token ids.
        if isinstance(req.prompt_formatted, str):
            token_ids = tokenizer.encode(
                req.prompt_formatted, add_special_tokens=False
            )
        elif isinstance(req.prompt_formatted, list):
            # Chat-formatted prompt — apply chat template.
            text = tokenizer.apply_chat_template(
                req.prompt_formatted, tokenize=False
            )
            token_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            logger.warning("Skipping request with unsupported prompt type")
            continue

        max_new = max_new_tokens_override or req.output_len or 128
        entries.append(
            WorkloadEntry(prompt_tokens=token_ids, max_new_tokens=max_new)
        )
    return entries


def sample_workload(
    *,
    dataset: str,
    tokenizer_name: str,
    num_prompts: int,
    seed: int,
    dataset_path: str | None = None,
    trust_remote_code: bool = False,
    max_new_tokens: int | None = None,
    random_input_len: str = "512",
    random_output_len: str = "128",
    sonnet_input_len: int = 512,
    sonnet_prefix_len: int = 128,
) -> list[WorkloadEntry]:
    """Sample and tokenize a workload from a shared dataset loader.

    Seeds NumPy and the ``random`` module from ``seed`` first, so a given
    ``(dataset, tokenizer, num_prompts, seed)`` reproduces the same bytes
    across the CLI and every runner that calls this.

    Args:
        dataset: Dataset name (sharegpt, sonnet, random, …).
        tokenizer_name: HuggingFace tokenizer id or path.
        num_prompts: Number of requests to sample.
        seed: Seed for the dataset sampler (and the RNGs it draws from).
        dataset_path: Local dataset file, overriding the HuggingFace download.
        trust_remote_code: Trust remote code when loading the tokenizer.
        max_new_tokens: Override ``max_new_tokens`` for every request. ``None``
            keeps each dataset's natural output length.
        random_input_len: Input length distribution for the random dataset.
        random_output_len: Output length distribution for the random dataset.
        sonnet_input_len: Input token length for the sonnet dataset.
        sonnet_prefix_len: Prefix token length for the sonnet dataset.

    Returns:
        The sampled workload entries, prompt tokens already resolved.
    """
    np.random.seed(seed)
    random.seed(seed)

    # Via ``get_tokenizer`` rather than ``AutoTokenizer`` so this matches what
    # ``TokenizerPool``'s workers build: the workers load through
    # ``get_tokenizer`` unconditionally, and it installs family-specific
    # ``encode`` overrides (Kimi K2.5 among them). Building the parent's copy
    # any other way leaves the pool measuring prompt lengths under different
    # rules than the single-threaded re-encode below. It also stashes the
    # resolved revision the pool pins its workers to.
    logger.info("Loading tokenizer: %s", tokenizer_name)
    tokenizer = get_tokenizer(
        tokenizer_name,
        revision=resolve_revision(tokenizer_name),
        trust_remote_code=trust_remote_code,
    )

    logger.info("Loading dataset: %s", dataset)
    benchmark_dataset = BenchmarkDataset.from_flags(
        dataset_name=dataset,
        dataset_path=dataset_path,
    )

    kwargs = build_dataset_kwargs(
        dataset,
        random_input_len=random_input_len,
        random_output_len=random_output_len,
        sonnet_input_len=sonnet_input_len,
        sonnet_prefix_len=sonnet_prefix_len,
    )
    logger.info("Sampling %d requests...", num_prompts)
    # ``random`` requires the pool and the others accept it, so it is
    # unconditional; workers spawn lazily, so a dataset that never encodes
    # through it pays nothing.
    with TokenizerPool(
        tokenizer,
        loader=functools.partial(
            worker_tokenizer_loader,
            trust_remote_code=trust_remote_code,
        ),
    ) as pool:
        samples = benchmark_dataset.sample_requests(
            num_requests=num_prompts,
            tokenizer=tokenizer,
            pool=pool,
            **kwargs,
        )

    logger.info("Tokenizing prompts...")
    return tokenize_requests(list(samples.requests), tokenizer, max_new_tokens)


def write_workload_file(
    path: Path,
    *,
    dataset: str,
    tokenizer_name: str,
    entries: list[WorkloadEntry],
    seed: int,
) -> None:
    """Write ``entries`` to ``path`` in the schema the harness consumes."""
    workload = {
        "metadata": {
            "dataset": dataset,
            "tokenizer": tokenizer_name,
            "num_requests": len(entries),
            "seed": seed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "requests": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(workload, f, separators=(",", ":"))
    logger.info("Wrote workload to %s", path)


def generate_workload_file(
    path: Path,
    *,
    dataset: str,
    tokenizer_name: str,
    num_prompts: int,
    seed: int,
    dataset_path: str | None = None,
    trust_remote_code: bool = False,
    max_new_tokens: int | None = None,
    random_input_len: str = "512",
    random_output_len: str = "128",
    sonnet_input_len: int = 512,
    sonnet_prefix_len: int = 128,
) -> list[WorkloadEntry]:
    """Sample a workload and write it to ``path``, returning the entries.

    Raises:
        ValueError: The sampler produced no usable requests, which would
            otherwise write an empty workload the harness rejects at load.
    """
    entries = sample_workload(
        dataset=dataset,
        tokenizer_name=tokenizer_name,
        num_prompts=num_prompts,
        seed=seed,
        dataset_path=dataset_path,
        trust_remote_code=trust_remote_code,
        max_new_tokens=max_new_tokens,
        random_input_len=random_input_len,
        random_output_len=random_output_len,
        sonnet_input_len=sonnet_input_len,
        sonnet_prefix_len=sonnet_prefix_len,
    )
    if not entries:
        raise ValueError(
            f"dataset {dataset!r} produced no valid requests for"
            f" {tokenizer_name!r}"
        )
    write_workload_file(
        path,
        dataset=dataset,
        tokenizer_name=tokenizer_name,
        entries=entries,
        seed=seed,
    )
    return entries
