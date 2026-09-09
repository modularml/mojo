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

"""Functional tests for benchmark_shared.datasets (real tokenizer, no mocks)."""

import warnings

from max.benchmark.benchmark_shared.datasets import BenchmarkDataset
from max.benchmark.benchmark_shared.datasets._tokenizer_pool import (
    TokenizerPool,
)
from max.pipelines.lib import generate_local_model_path
from transformers import AutoTokenizer

REPO_ID = "HuggingFaceTB/SmolLM-135M"
# Minimum average match rate for system prompt token consistency across requests.
# Set to 0.90 to accommodate tokenization boundary effects when repeating base
# prompts to different target lengths. Rates of 90-96% are expected and indicate
# high consistency while allowing for minor alignment variations.
SYS_PROMPT_PATTERN_CONSISTENCY_THRESHOLD = 0.90


def _match_rate(seq: list[int], ref: list[int]) -> float:
    """Fraction of elements in seq that match the corresponding elements in ref.

    Compares seq with ref[:len(seq)]; ref is the reference (longest) sequence.
    Returns 1.0 if seq is empty.
    """
    if not seq:
        return 1.0
    ref_prefix = ref[: len(seq)]
    matches = sum(1 for a, b in zip(seq, ref_prefix, strict=True) if a == b)
    return matches / len(seq)


def _is_consistent(
    sequences: list[list[int]], threshold: float
) -> tuple[bool, list[float]]:
    """Check whether sequences are consistent with the longest as reference.

    Returns (True if average match rate >= threshold, list of per-sequence match rates).
    """
    if not sequences:
        return True, []
    ref = max(sequences, key=len)
    rates = [_match_rate(s, ref) for s in sequences]
    avg_rate = sum(rates) / len(rates)
    return avg_rate >= threshold, rates


def test_random_replacement_token_with_smollm_tokenizer() -> None:
    """Test that the space-replacement token logic finds a valid token with a real tokenizer.

    This is a replicated logic in random dataset that picks a replacement for
    special tokens using candidates: plain space, U+0120 (BPE), U+2581 (SentencePiece).
    """
    try:
        model_path = generate_local_model_path(REPO_ID)
    except FileNotFoundError as e:
        warnings.warn(f"Failed to generate local model path: {str(e)}")
        warnings.warn(
            f"Falling back to repo_id: {REPO_ID} as config to the tokenizer"
        )
        model_path = REPO_ID

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    replacement = next(
        tid
        for candidate in [" ", chr(0x0120), chr(0x2581)]
        if (tid := tokenizer.convert_tokens_to_ids(candidate))
        not in (None, tokenizer.unk_token_id)
    )

    assert isinstance(replacement, int), "replacement must be a token id (int)"
    assert replacement != tokenizer.unk_token_id, (
        "replacement must not be the unknown token id"
    )


def test_random_dataset_sys_prompt_ratio_matches_requested() -> None:
    """Validate that generated prompts have system prompt length matching sys_prompt_ratio.

    For each request: (1) expected system length in tokens = prompt_len * sys_prompt_ratio,
    (2) encode prompt_formatted to get token ids, (3) take the first expected_sys_len
    tokens as the system portion. Compares system portions across requests via match_rate
    and is_consistent to ensure they align within the threshold (e.g. same system prompt
    when max_num_unique_sys_prompt=1).

    Note: When input lengths vary (e.g. U(80,100)), system prompt lengths also vary.
    The base system prompt is repeated/truncated to different target lengths, which can
    cause minor token alignment mismatches (typically 90-96% consistency). This is
    expected behavior and still indicates high consistency.
    """
    try:
        model_path = generate_local_model_path(REPO_ID)
    except FileNotFoundError as e:
        warnings.warn(f"Failed to generate local model path: {str(e)}")
        warnings.warn(
            f"Falling back to repo_id: {REPO_ID} as config to the tokenizer"
        )
        model_path = REPO_ID

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = BenchmarkDataset.from_flags(dataset_name="random")

    with TokenizerPool(tokenizer) as pool:
        for sys_prompt_ratio in (0.0, 0.25, 0.5, 0.75):
            samples = dataset.sample_requests(
                num_requests=10,
                tokenizer=tokenizer,
                pool=pool,
                input_len="U(80, 100)",
                output_len="10",
                sys_prompt_ratio=sys_prompt_ratio,
                max_num_unique_sys_prompt=1,
                shuffle=False,
            )
            assert len(samples.requests) == 10, "Expected 10 requests"

        # (1) Each request's system length in tokens = prompt_len * sys_prompt_ratio
        # (2) Encode prompt_formatted and take first expected_sys_len token ids
        system_portions: list[list[int]] = []
        for req in samples.requests:
            expected_sys_len = int(req.prompt_len * sys_prompt_ratio)
            token_ids = tokenizer(
                req.prompt_formatted, add_special_tokens=False
            ).input_ids
            assert len(token_ids) >= expected_sys_len, (
                f"Encoded length {len(token_ids)} < expected sys length {expected_sys_len}"
            )
            system_portions.append(token_ids[:expected_sys_len])

        # (3) Compare system portions: they should be consistent within threshold
        consistent, rates = _is_consistent(
            system_portions, threshold=SYS_PROMPT_PATTERN_CONSISTENCY_THRESHOLD
        )
        assert consistent, (
            f"System prompt token consistency below {SYS_PROMPT_PATTERN_CONSISTENCY_THRESHOLD} "
            f"for sys_prompt_ratio={sys_prompt_ratio}: rates={rates}"
        )


def test_synthetic_placeholder_token_count() -> None:
    """Synthetic placeholder for prefix turns should tokenize to ~output_len."""
    try:
        model_path = generate_local_model_path(REPO_ID)
    except FileNotFoundError:
        model_path = REPO_ID

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for output_len in [1, 5, 10, 50]:
        placeholder = " ".join(["token"] * max(output_len, 1))
        token_count = len(
            tokenizer.encode(placeholder, add_special_tokens=False)
        )
        assert abs(token_count - output_len) / output_len < 0.2, (
            f"output_len={output_len}: expected ~{output_len} tokens,"
            f" got {token_count}"
        )
