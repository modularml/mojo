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

import argparse
import os
import re
import sys

_RE = re.compile(r"\b(?:XFAIL|REQUIRES|UNSUPPORTED):\s*(.+)\s*$", re.MULTILINE)
_KNOWN_FEATURES = {
    "*",  # Alias for disabling with XFAIL
    "asan",
    "DISABLED",
    "disabled",
    "emit-mojo",
    "manual",  # Alias for disabling
    "msan",
    "macos-26+",
    "production",
    "system-darwin",
    "system-linux",
    "tsan",
    "ubsan",
    "x86_64-linux",
    "ASSERTIONS",
    "air-objdump",
}

_GPU_FEATURES = {
    "A10-GPU",
    "A100-GPU",
    "AMD-GPU",
    "APPLE-GPU",
    "B200-GPU",
    "H100-GPU",
    "MI300X-GPU",
    "NVIDIA-GPU",
}

_FEATURES_BY_PREFIX = {
    "max/kernels/test/linalg": {
        "avx2",
        "intel_amx",
        "neon_dotprod",
        "neon_matmul",
    },
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("files", nargs="+")
    return parser


def _get_unsupported_features(
    root: str, contents: str, include_gpu_features: bool
) -> set[str]:
    features = _RE.findall(contents)
    if not features:
        return set()

    used_features = set()
    for feature in features:
        used_features |= set(
            feature.replace("|", " ")
            .replace("&", " ")
            .replace(",", " ")
            .replace("!", " ")
            .split(" ")
        )
    used_features.discard("")  # Empty spaces from split

    valid_features = _KNOWN_FEATURES | _FEATURES_BY_PREFIX.get(root, set())
    if include_gpu_features:
        valid_features |= _GPU_FEATURES

    return used_features - valid_features


def _has_run_command(contents: str) -> bool:
    return "RUN:" in contents


def _main(root: str, files: list[str]) -> None:
    include_gpu_features = os.environ["RUNS_ON_GPU"] == "True"
    errors = []
    for file in files:
        path = os.path.join(root, file)
        with open(path) as f:
            contents = f.read()

        if not _has_run_command(contents):
            errors.append(
                f"error: {path} does not have any RUN commands, exclude it from the lit_tests or add commands"
            )

        if invalid_features := _get_unsupported_features(
            root, contents, include_gpu_features
        ):
            errors.append(
                f"error: {path} contains invalid lit features:"
                f" {', '.join(sorted(invalid_features))}"
            )

    if errors:
        print("\n".join(sorted(errors)))
        sys.exit(1)


if __name__ == "__main__":
    args = _build_parser().parse_args()
    _main(args.root, args.files)
