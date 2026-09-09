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

# This python script is used to parse matmul autotune configs from cutlass_profiler
# and retrieve the best config according to "Runtime" from all the configs which
# have been explored by cutlass_profiler on a matmul shape. Each matmul shape
# should have a csv output from cutlass_profiler.
# Usage:
#    python3 cutlass_profiler_parser.py -k <kernel_string> -m <mode> \
#        -o <output=output.csv> -s <split_k_slices> -d M N K

import argparse
import subprocess
from argparse import Namespace
from collections import defaultdict

import pandas as pd


def retrieve_data(files: list[str], output: str) -> None:
    res = []
    df = pd.read_csv(fname)
    df = df.sort_values(by="Runtime")
    res.append(df.iloc[0])
    res_df = pd.DataFrame(res)
    res_df.to_csv(output, index=False, header=True)


def build_cmd(args: Namespace, args_map: defaultdict(list)) -> str:
    cmd = ["./tools/profiler/cutlass_profiler"]
    cmd.append(f"--operation=={args.operation}")
    for key, val in vars(args).items():
        if key in args_map:
            cmd.append(f"{args_map[key]}={val}")
    cmd.append(f"--m={args.dims[0]} --n={args.dims[1]} --k={args.dims[2]}")
    cmd.append(
        f"--output=matmul_M{args.dims[0]}_N{args.dims[1]}_K{args.dims[2]}"
    )
    return " ".join(cmd)


def run_cmd(cmd: str) -> None:
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(e.stdout.decode("utf-8"))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="This script is used to parse cutlass profiler results"
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="serial",
        choices=["serial", "parallel"],
        help="Split K mode (parallel|serial)",
    )
    parser.add_argument(
        "-k",
        "--kernels",
        help="Kernels to be profiled by cutlass",
    )
    parser.add_argument(
        "-d",
        "--dims",
        nargs=3,
        type=int,
        default=[1, 1, 1],
        help="The matmul shape M N K",
    )
    parser.add_argument(
        "-p",
        "--operation",
        default="Gemm",
        help="Operation",
    )
    parser.add_argument(
        "-s",
        "--splits",
        default=1,
        help="Split K partitions",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.csv",
        help="csv file stores the best config",
    )
    args = parser.parse_args()
    prefix = f"matmul_M{args.dims[0]}_N{args.dims[1]}_K{args.dims[2]}"
    dump_file = args.output or prefix + "_config.csv"

    args_map = defaultdict(str)
    args_map["kernels"] = "--kernels"
    args_map["mode"] = "--split_k_mode"
    args_map["splits"] = "--split_k_slices"

    # First run cutlass_profiler on a matmul shape
    run_cmd(build_cmd(args, args_map))

    # Get the best config which has the smallest Runtime.
    retrieve_data(prefix + ".gemm.csv", dump_file)


if __name__ == "__main__":
    main()
