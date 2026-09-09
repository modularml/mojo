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

"""Parameter sweep script for MHA decode benchmarks.

Runs benchmarks across multiple parameter combinations and outputs results to CSV.

Usage:
    # Via Bazel
    ./bazelw run //Kernels/benchmarks/comparison:sweep_decode -- --output results.csv

    # Direct Python (with PYTHONPATH set)
    PYTHONPATH=/home/ubuntu/modular/Kernels/utils python3 sweep_decode.py --output results.csv

    # Custom parameter sweep
    python3 sweep_decode.py --output results.csv \\
        --batch-sizes 32 64 128 \\
        --cache-lens 512 1024 2048 \\
        --num-heads 128 \\
        --num-kv-heads 8
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from bench_blackwell_decode import (
    bench_flashinfer,
    bench_max,
)


def run_sweep(
    batch_sizes: list[int],
    cache_lens: list[int],
    num_heads_list: list[int],
    num_kv_heads_list: list[int],
    head_dims: list[int],
    dtypes: list[str],
    output_file: Path,
    skip_flashinfer: bool = False,
    skip_max: bool = False,
) -> None:
    """Run parameter sweep and save results to CSV.

    Args:
        batch_sizes: List of batch sizes to test
        cache_lens: List of cache lengths to test
        num_heads_list: List of query head counts to test
        num_kv_heads_list: List of KV head counts to test
        head_dims: List of head dimensions to test
        dtypes: List of dtypes to test (e.g., ["bfloat16", "float16"])
        output_file: Output CSV file path
        skip_flashinfer: Skip FlashInfer benchmarks
        skip_max: Skip MAX benchmarks
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    # Generate all combinations
    combinations = list(
        itertools.product(
            batch_sizes,
            cache_lens,
            num_heads_list,
            num_kv_heads_list,
            head_dims,
            dtypes,
        )
    )

    print(f"Running {len(combinations)} benchmark configurations...")
    print(f"Output will be saved to: {output_file}")
    print("=" * 80)

    # Prepare CSV output with one row per config
    fieldnames = [
        "batch_size",
        "cache_len",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "dtype",
        "time_ms (flashinfer)",
        "gb_per_sec (flashinfer)",
        "time_ms (max)",
        "gb_per_sec (max)",
        "speedup (max/flashinfer)",
    ]

    results: list[dict[str, Any]] = []

    # Run benchmarks
    for idx, (
        batch_size,
        cache_len,
        num_heads,
        num_kv_heads,
        head_dim,
        dtype_str,
    ) in enumerate(combinations, 1):
        print(
            f"[{idx}/{len(combinations)}] "
            f"batch={batch_size}, cache={cache_len}, "
            f"q_heads={num_heads}, kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}, dtype={dtype_str}"
        )

        dtype_torch = dtype_map[dtype_str]

        # Initialize row for this configuration
        row = {
            "batch_size": batch_size,
            "cache_len": cache_len,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "dtype": dtype_str,
            "time_ms (flashinfer)": None,
            "gb_per_sec (flashinfer)": None,
            "time_ms (max)": None,
            "gb_per_sec (max)": None,
            "speedup (max/flashinfer)": None,
        }

        # Benchmark FlashInfer
        if not skip_flashinfer:
            try:
                result = bench_flashinfer(
                    batch_size,
                    cache_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    64,  # TRTLLM's page table size
                    dtype_torch,
                    use_tensor_cores=True,
                    backend="trtllm",
                )
                if result is not None:
                    time_s, gb_per_sec = result
                    row["time_ms (flashinfer)"] = time_s * 1000
                    row["gb_per_sec (flashinfer)"] = gb_per_sec
                    print(
                        f"  FlashInfer: {time_s * 1000:.4f} ms, {gb_per_sec:.2f} GB/s"
                    )
                else:
                    print("  FlashInfer: Not available")
            except Exception as e:
                print(f"  FlashInfer: Failed - {e}")

        # Benchmark MAX
        if not skip_max:
            try:
                result = bench_max(
                    batch_size,
                    cache_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    128,  # MAX's page size
                    dtype_torch,
                )
                if result is not None:
                    time_s, gb_per_sec = result
                    row["time_ms (max)"] = time_s * 1000
                    row["gb_per_sec (max)"] = gb_per_sec
                    print(
                        f"  MAX:        {time_s * 1000:.4f} ms, {gb_per_sec:.2f} GB/s"
                    )
                else:
                    print("  MAX:        Not available")
            except Exception as e:
                print(f"  MAX:        Failed - {e}")

        # Calculate speedup (max/flashinfer) if both results are available
        max_time = row["time_ms (max)"]
        flashinfer_time = row["time_ms (flashinfer)"]
        if max_time is not None and flashinfer_time is not None:
            # Type assertions for mypy
            assert isinstance(max_time, (int, float))
            assert isinstance(flashinfer_time, (int, float))
            speedup = max_time / flashinfer_time
            row["speedup (max/flashinfer)"] = speedup
            if speedup < 1.0:
                print(f"  → MAX is {1 / speedup:.2f}x faster than FlashInfer")
            else:
                print(f"  → FlashInfer is {speedup:.2f}x faster than MAX")

        results.append(row)
        print()

    # Write results to CSV
    print("=" * 80)
    print(f"Writing results to {output_file}...")

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"✓ Saved {len(results)} configurations to {output_file}")

    # Print summary
    print("\nSummary:")
    flashinfer_success = sum(
        1 for r in results if r["time_ms (flashinfer)"] is not None
    )
    flashinfer_failed = sum(
        1
        for r in results
        if r["time_ms (flashinfer)"] is None and not skip_flashinfer
    )
    max_success = sum(1 for r in results if r["time_ms (max)"] is not None)
    max_failed = sum(
        1 for r in results if r["time_ms (max)"] is None and not skip_max
    )

    if not skip_flashinfer:
        print(
            f"  FlashInfer: {flashinfer_success} success, {flashinfer_failed} failed"
        )
    if not skip_max:
        print(f"  MAX: {max_success} success, {max_failed} failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter sweep for MHA decode benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Llama 3.1 70B decode sweep
  python3 sweep_decode.py --output llama31_decode.csv \\
      --batch-sizes 1 2 4 8 16 32 \\
      --cache-lens 512 1024 2048 4096 \\
      --num-heads 128 \\
      --num-kv-heads 8 \\
      --head-dims 128 \\
      --dtypes bfloat16

  # Multi-configuration sweep
  python3 sweep_decode.py --output multi_config.csv \\
      --batch-sizes 32 64 128 \\
      --cache-lens 1024 2048 \\
      --num-heads 32 64 128 \\
      --num-kv-heads 4 8 \\
      --head-dims 64 128 \\
      --dtypes bfloat16 float16
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output CSV file path (relative to current directory or absolute)",
    )

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="Batch sizes to test (default: 1 2 4 8 16 32)",
    )

    parser.add_argument(
        "--cache-lens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="Cache lengths to test (default: 512 1024 2048 4096)",
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        nargs="+",
        default=[128],
        help="Number of query heads to test (default: 128)",
    )

    parser.add_argument(
        "--num-kv-heads",
        type=int,
        nargs="+",
        default=[8],
        help="Number of KV heads to test (default: 8)",
    )

    parser.add_argument(
        "--head-dims",
        type=int,
        nargs="+",
        default=[128],
        help="Head dimensions to test (default: 128)",
    )

    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        choices=["float16", "bfloat16", "float32"],
        default=["bfloat16"],
        help="Data types to test (default: bfloat16)",
    )

    parser.add_argument(
        "--skip-flashinfer",
        action="store_true",
        help="Skip FlashInfer benchmarks",
    )

    parser.add_argument(
        "--skip-max",
        action="store_true",
        help="Skip MAX benchmarks",
    )

    args = parser.parse_args()

    # Resolve output path to be relative to invocation directory (not Bazel runfiles)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        # When running via Bazel, BUILD_WORKING_DIRECTORY contains the directory
        # where the user invoked bazel. Use that instead of cwd.
        working_dir = os.environ.get("BUILD_WORKING_DIRECTORY")
        if working_dir:
            output_path = Path(working_dir) / output_path
        else:
            # Running directly with Python, use current directory
            output_path = Path.cwd() / output_path

    # Ensure the output path is absolute
    output_path = output_path.resolve()

    # Print configuration
    print("=" * 80)
    print("MHA Decode Parameter Sweep")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Output file: {output_path}")
    print()
    print("Parameters:")
    print(f"  Batch sizes:    {args.batch_sizes}")
    print(f"  Cache lengths:  {args.cache_lens}")
    print(f"  Num heads:      {args.num_heads}")
    print(f"  Num KV heads:   {args.num_kv_heads}")
    print(f"  Head dims:      {args.head_dims}")
    print(f"  Dtypes:         {args.dtypes}")
    print()
    print(f"  Skip FlashInfer: {args.skip_flashinfer}")
    print(f"  Skip MAX:        {args.skip_max}")
    print()

    total_configs = (
        len(args.batch_sizes)
        * len(args.cache_lens)
        * len(args.num_heads)
        * len(args.num_kv_heads)
        * len(args.head_dims)
        * len(args.dtypes)
    )
    num_implementations = 2 - int(args.skip_flashinfer) - int(args.skip_max)
    print(f"Total configurations: {total_configs}")
    print(f"Benchmarks per config: {num_implementations}")
    print(f"Total benchmark runs: {total_configs * num_implementations}")
    print("=" * 80)
    print()

    run_sweep(
        batch_sizes=args.batch_sizes,
        cache_lens=args.cache_lens,
        num_heads_list=args.num_heads,
        num_kv_heads_list=args.num_kv_heads,
        head_dims=args.head_dims,
        dtypes=args.dtypes,
        output_file=output_path,
        skip_flashinfer=args.skip_flashinfer,
        skip_max=args.skip_max,
    )


if __name__ == "__main__":
    main()
