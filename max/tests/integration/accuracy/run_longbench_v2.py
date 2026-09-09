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

"""LongBench v2 evaluator for pipeline dataset evaluation.

This implementation closely follows the official LongBench v2 evaluation script:
https://github.com/THUDM/LongBench/blob/main/pred.py

LongBench v2 is a benchmark for evaluating LLMs' ability to handle long-context
problems requiring deep understanding and reasoning across real-world multitasks.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import re
import time
from pathlib import Path
from typing import Any

import click
from datasets import Dataset, load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger("longbench_v2_eval")

# Prompt templates from:
# https://github.com/THUDM/LongBench/tree/main/prompts
TEMPLATE_0SHOT = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""

TEMPLATE_0SHOT_COT = """Please read the following text and answer the questions below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Let's think step by step:"""

TEMPLATE_0SHOT_COT_ANS = """Please read the following text and answer the questions below.

The text is too long and omitted here.

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Let's think step by step: $COT$

Based on the above, what is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)"."""

TEMPLATE_0SHOT_NO_CONTEXT = """What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

What is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)"."""


def query_llm(
    prompt: str,
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    client: OpenAI,
    max_len: int,
    temperature: float = 0.5,
    max_new_tokens: int = 128,
) -> str:
    """Query the LLM.

    Args:
        prompt: The prompt to send.
        model: Model name.
        tokenizer: Tokenizer for the model.
        client: OpenAI client.
        max_len: Maximum tokens allowed for the prompt content.
        temperature: Sampling temperature.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        Model response text.
    """
    # Truncate prompt to fit context window
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) > max_len:
        input_ids = input_ids[: max_len // 2] + input_ids[-max_len // 2 :]
        prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    tries = 0
    while tries < 5:
        tries += 1
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            return completion.choices[0].message.content or ""
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Error Occurs: {e}        Retry ...")
            time.sleep(1)
    print("Max tries. Failed.")
    return ""


def extract_answer(response: str) -> str | None:
    """Extract answer from model response.
    https://github.com/THUDM/LongBench/blob/main/pred.py

    Args:
        response: The model's response text.

    Returns:
        The extracted answer letter (A-D), or None if not found.
    """
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    else:
        match = re.search(r"The correct answer is ([A-D])", response)
        if match:
            return match.group(1)
        else:
            return None


def get_pred(
    item: dict[str, Any],
    model: str,
    client: OpenAI,
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
    cot: bool = False,
    no_context: bool = False,
    max_new_tokens: int = 128,
    cot_max_new_tokens: int = 1024,
) -> dict[str, Any]:
    """Get prediction for a single item.

    Args:
        item: Dataset item.
        model: Model name.
        client: OpenAI client.
        tokenizer: Tokenizer for the model.
        max_len: Maximum tokens allowed for prompt content.
        cot: Enable chain-of-thought.
        no_context: Omit context.
        max_new_tokens: Maximum new tokens to generate for non-CoT responses.
        cot_max_new_tokens: Maximum new tokens for CoT reasoning.

    Returns:
        Result dictionary with prediction and metadata.
    """
    if no_context:
        template = TEMPLATE_0SHOT_NO_CONTEXT
    elif cot:
        template = TEMPLATE_0SHOT_COT
    else:
        template = TEMPLATE_0SHOT
    context = item["context"]
    prompt = template.replace("$DOC$", context.strip())
    prompt = prompt.replace("$Q$", item["question"].strip())
    prompt = prompt.replace("$C_A$", item["choice_A"].strip())
    prompt = prompt.replace("$C_B$", item["choice_B"].strip())
    prompt = prompt.replace("$C_C$", item["choice_C"].strip())
    prompt = prompt.replace("$C_D$", item["choice_D"].strip())
    if cot:
        output = query_llm(
            prompt,
            model,
            tokenizer,
            client,
            max_len,
            temperature=0.1,
            max_new_tokens=cot_max_new_tokens,
        )
    else:
        output = query_llm(
            prompt,
            model,
            tokenizer,
            client,
            max_len,
            temperature=0.1,
            max_new_tokens=max_new_tokens,
        )
    if output == "":
        return item
    if cot:
        response = output.strip()
        item["response_cot"] = response
        prompt = TEMPLATE_0SHOT_COT_ANS.replace("$DOC$", context.strip())
        prompt = prompt.replace("$Q$", item["question"].strip())
        prompt = prompt.replace("$C_A$", item["choice_A"].strip())
        prompt = prompt.replace("$C_B$", item["choice_B"].strip())
        prompt = prompt.replace("$C_C$", item["choice_C"].strip())
        prompt = prompt.replace("$C_D$", item["choice_D"].strip())
        prompt = prompt.replace("$COT$", response)
        output = query_llm(
            prompt,
            model,
            tokenizer,
            client,
            max_len,
            temperature=0.1,
            max_new_tokens=128,
        )
        if output == "":
            return item
    response = output.strip()
    item["response"] = response
    item["pred"] = extract_answer(response)
    item["judge"] = item["pred"] == item["answer"]
    item["context"] = context[:1000]
    return item


def calculate_metrics(
    results: list[dict[str, Any]],
    compensated: bool = False,
) -> dict[str, Any]:
    """Calculate accuracy metrics by different groupings.

    Ref: https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/result.py

    Args:
        results: List of result dictionaries.
        compensated: If True, treat None predictions as 0.25 accuracy
            (random guess among 4 choices).

    Returns:
        Dictionary with overall and grouped metrics.
    """
    total = len(results)

    # Counters for difficulty groups
    easy_count, hard_count = 0, 0
    easy_acc, hard_acc = 0.0, 0.0

    # Counters for length groups
    short_count, medium_count, long_count = 0, 0, 0
    short_acc, medium_acc, long_acc = 0.0, 0.0, 0.0

    for r in results:
        acc = float(r.get("judge", False))
        if compensated and r.get("pred") is None:
            acc = 0.25

        # Group by difficulty
        if r.get("difficulty") == "easy":
            easy_count += 1
            easy_acc += acc
        else:
            hard_count += 1
            hard_acc += acc

        # Group by length
        length = r.get("length", "")
        if length == "short":
            short_count += 1
            short_acc += acc
        elif length == "medium":
            medium_count += 1
            medium_acc += acc
        else:
            long_count += 1
            long_acc += acc

    overall_acc = (easy_acc + hard_acc) / total if total > 0 else 0.0

    return {
        "accuracy": overall_acc,
        "total": total,
        "by_difficulty": {
            "easy": {
                "accuracy": easy_acc / easy_count if easy_count > 0 else 0.0,
                "correct": easy_acc,
                "total": easy_count,
            },
            "hard": {
                "accuracy": hard_acc / hard_count if hard_count > 0 else 0.0,
                "correct": hard_acc,
                "total": hard_count,
            },
        },
        "by_length": {
            "short": {
                "accuracy": short_acc / short_count if short_count > 0 else 0.0,
                "correct": short_acc,
                "total": short_count,
            },
            "medium": {
                "accuracy": medium_acc / medium_count
                if medium_count > 0
                else 0.0,
                "correct": medium_acc,
                "total": medium_count,
            },
            "long": {
                "accuracy": long_acc / long_count if long_count > 0 else 0.0,
                "correct": long_acc,
                "total": long_count,
            },
        },
    }


def _worker_process(
    rank: int,
    data_subset: list[dict[str, Any]],
    model: str,
    url: str,
    client_timeout: int,
    max_len: int,
    cot: bool,
    no_context: bool,
    max_new_tokens: int,
    cot_max_new_tokens: int,
    output_file: Path,
) -> None:
    """Worker process for parallel evaluation.

    Args:
        rank: Process rank for progress display.
        data_subset: Subset of dataset items to process.
        model: Model name.
        url: API base URL.
        client_timeout: Client timeout.
        max_len: Maximum tokens allowed for prompt content.
        cot: Enable chain-of-thought.
        no_context: Omit context.
        max_new_tokens: Maximum new tokens to generate for non-CoT responses.
        cot_max_new_tokens: Maximum new tokens for CoT reasoning.
        output_file: Path to output file for writing results.
    """
    # Each worker creates its own client and tokenizer
    client = OpenAI(
        base_url=url,
        api_key="token-abc123",
        timeout=client_timeout,
    )
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tokenizer.model_max_length = max_len

    with open(output_file, "a") as fout:
        for item in tqdm(data_subset, desc=f"Worker {rank}"):
            try:
                result = get_pred(
                    dict(item),
                    model,
                    client,
                    tokenizer,
                    max_len,
                    cot=cot,
                    no_context=no_context,
                    max_new_tokens=max_new_tokens,
                    cot_max_new_tokens=cot_max_new_tokens,
                )
            except Exception as e:
                result = {
                    "_id": item["_id"],
                    "domain": item["domain"],
                    "sub_domain": item.get("sub_domain", ""),
                    "difficulty": item.get("difficulty", ""),
                    "length": item.get("length", ""),
                    "question": item["question"],
                    "pred": None,
                    "answer": item["answer"],
                    "judge": False,
                    "response": f"ERROR: {e}",
                }
            fout.write(json.dumps(result) + "\n")
            fout.flush()


@click.command()
@click.option(
    "--save_dir",
    "-s",
    type=str,
    default="results",
    help="Directory to save results",
)
@click.option(
    "--model",
    "-m",
    type=str,
    required=True,
    help="Model name for evaluation",
)
@click.option(
    "--cot",
    "-cot",
    is_flag=True,
    help="Enable chain-of-thought prompting",
)
@click.option(
    "--no_context",
    "-nc",
    is_flag=True,
    help="Omit context (measure memorization)",
)
@click.option(
    "--url",
    type=str,
    default="http://127.0.0.1:8000/v1",
    help="API base URL",
)
@click.option(
    "--max_samples",
    type=int,
    default=None,
    help="Limit number of samples (for testing)",
)
@click.option(
    "--max_context_length",
    type=int,
    default=None,
    help="Override max context length",
)
@click.option(
    "--client_timeout",
    type=int,
    default=1000,
    help="Client timeout",
)
@click.option(
    "--n_proc",
    type=int,
    default=16,
    help="Number of parallel processes for evaluation",
)
@click.option(
    "--max_new_tokens",
    type=int,
    default=128,
    help="Maximum new tokens to generate for non-CoT responses",
)
@click.option(
    "--cot_max_new_tokens",
    type=int,
    default=1024,
    help="Maximum new tokens to generate for CoT reasoning",
)
def main(
    save_dir: str,
    model: str,
    cot: bool,
    no_context: bool,
    url: str,
    max_samples: int | None,
    max_context_length: int | None,
    client_timeout: int,
    n_proc: int,
    max_new_tokens: int,
    cot_max_new_tokens: int,
) -> None:
    """Run LongBench v2 evaluation.

    This implementation follows the official LongBench v2 evaluation:
    https://github.com/THUDM/LongBench/blob/main/pred.py
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(name)s: %(message)s",
    )

    # Load dataset
    logger.info("Loading LongBench v2 dataset...")
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    assert isinstance(dataset, Dataset), "Expected Dataset, got different type"
    logger.info(f"Loaded {len(dataset)} samples")

    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        logger.info(f"Limited to {len(dataset)} samples")

    # Load model config to get context length
    logger.info(f"Loading config for {model}...")
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    model_max_length = getattr(
        config, "max_position_embeddings", None
    ) or getattr(
        getattr(config, "text_config", None), "max_position_embeddings", None
    )
    context_length = max_context_length or model_max_length

    logger.info(
        f"Context length: {context_length} (model max: {model_max_length}), "
    )

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    results: list[dict[str, Any]] = []
    logger.info(f"Starting evaluation with {n_proc} process(es)...")

    data_list = list(dataset)
    data_subsets = [data_list[i::n_proc] for i in range(n_proc)]

    temp_results_file = output_dir / "longbench_v2_temp_results.jsonl"
    if temp_results_file.exists():
        temp_results_file.unlink()

    processes = []
    for rank in range(n_proc):
        p = mp.Process(
            target=_worker_process,
            args=(
                rank,
                data_subsets[rank],
                model,
                url,
                client_timeout,
                context_length,
                cot,
                no_context,
                max_new_tokens,
                cot_max_new_tokens,
                temp_results_file,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    with open(temp_results_file) as f:
        for line in f:
            results.append(json.loads(line.strip()))

    temp_results_file.unlink()

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Prepare output
    output = {
        "model_name": model,
        "evaluator": "longbench-v2",
        "config": {
            "cot": cot,
            "no_context": no_context,
        },
        "metrics": metrics,
    }

    # Save results
    output_file = output_dir / "longbench_v2_results.json"

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("LongBench v2 Evaluation Results")
    print(f"{'=' * 60}")
    print(f"Model: {model}")
    print(f"COT: {cot}, No Context: {no_context}")
    print(
        f"Overall Accuracy: {100 * metrics['accuracy']:.1f}% "
        f"({metrics['total']} samples)"
    )
    print("\nBy Difficulty:")
    for diff in ["easy", "hard"]:
        m = metrics["by_difficulty"][diff]
        print(f"  {diff}: {100 * m['accuracy']:.1f}% ({m['total']} samples)")
    print("\nBy Length:")
    for length in ["short", "medium", "long"]:
        m = metrics["by_length"][length]
        print(f"  {length}: {100 * m['accuracy']:.1f}% ({m['total']} samples)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
