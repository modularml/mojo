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

"""Utilities for exploring supported pipelines."""

import json

from max.pipelines import PIPELINE_REGISTRY


def list_pipelines_to_console() -> None:
    print()
    # Print human readable format
    for arch in PIPELINE_REGISTRY.all_architectures():
        print()
        print(f"    Architecture: {arch.name}")
        print()
        print("         Example Huggingface Repo Ids: ")
        for repo_id in arch.example_repo_ids:
            print(f"              {repo_id}")

        print()
        for encoding_name in arch.supported_encodings:
            print(f"         Encoding Supported: {encoding_name}")

    print()


def list_pipelines_to_json() -> None:
    """Print the list of pipelines architecture options in JSON format."""
    architectures = {}
    for arch in PIPELINE_REGISTRY.all_architectures():
        architectures[arch.name] = {
            "example_repo_ids": list(arch.example_repo_ids),
            "supported_encodings": list(arch.supported_encodings),
        }
    print(json.dumps({"architectures": architectures}, indent=2))
