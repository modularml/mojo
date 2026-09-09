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
"""Generate http_file entries in common.MODULE.bazel from logit_verification_config.yaml. A script to run manually"""

import os
import re
from pathlib import Path

from max.tests.integration.accuracy.logit_verification.logit_verification_config import (
    LOGIT_VERIFICATION_CONFIG,
)

_WORKSPACE = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", "."))

MODULE_PATH = _WORKSPACE / "oss/modular/bazel/common.MODULE.bazel"

# Markers to identify the generated section in MODULE.bazel
BEGIN_MARKER = "# BEGIN GENERATED LOGIT_VERIFICATION GOLDEN HTTP_FILES"
END_MARKER = "# END GENERATED LOGIT_VERIFICATION GOLDEN HTTP_FILES"


def s3_to_https(s3_url: str) -> str:
    """Convert s3://bucket/key to https://bucket.s3.amazonaws.com/key."""
    s3_url = s3_url.removeprefix("s3://")
    bucket, _, key = s3_url.partition("/")
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def extract_sha256_from_url(url: str) -> str:
    """Extract sha256 from the URL path (convention in this repo: .../sha256/filename)."""
    # URLs follow: .../artifacts/name/version/sha256/filename.tar.gz
    parts = url.rstrip("/").split("/")
    return parts[-2]


def generate_golden_entry(name: str, sha256: str, url: str) -> str:
    return f'    {{"name": "{name}", "sha256": "{sha256}", "url": "{url}"}},'


def generate_section(entries: list[tuple[str, str, str]]) -> str:
    entry_lines = "\n".join(
        generate_golden_entry(name, sha256, url)
        for name, sha256, url in entries
    )
    return (
        f"{BEGIN_MARKER}\n"
        f"\n"
        f"LOGIT_VERIFICATION_GOLDENS = [\n"
        f"{entry_lines}\n"
        f"]\n"
        f"\n"
        f"[\n"
        f"    http_file(\n"
        f'        name = entry["name"],\n'
        f'        sha256 = entry["sha256"],\n'
        f'        url = entry["url"],\n'
        f"    )\n"
        f"    for entry in LOGIT_VERIFICATION_GOLDENS\n"
        f"]\n"
        f"\n"
        f"{END_MARKER}"
    )


def main() -> None:
    # Collect unique tar_files (multiple pipelines may share the same tar)
    seen: dict[str, tuple[str, str, str]] = {}  # name -> (name, sha256, url)
    for pipeline_config in LOGIT_VERIFICATION_CONFIG.pipelines.values():
        goldens = pipeline_config.pregenerated_torch_goldens
        if not goldens:
            continue
        https_url = s3_to_https(goldens.tar_file)
        sha256 = extract_sha256_from_url(https_url)
        name = https_url.split("/")[-1]  # e.g. torch_olmo-1b-hf_golden.tar.gz
        if name not in seen:
            seen[name] = (name, sha256, https_url)

    entries: list[tuple[str, str, str]] = list(seen.values())
    generated = generate_section(entries)

    module_content = MODULE_PATH.read_text()

    if BEGIN_MARKER in module_content:
        # Replace existing generated section
        pattern = re.compile(
            rf"{re.escape(BEGIN_MARKER)}.*?{re.escape(END_MARKER)}",
            re.DOTALL,
        )
        new_content = pattern.sub(generated, module_content)
    else:
        # Append before the first non-golden http_file or at end of file
        new_content = module_content + "\n" + generated + "\n"

    MODULE_PATH.write_text(new_content)
    print(f"Updated {MODULE_PATH} with {len(seen)} golden http_file entries.")


if __name__ == "__main__":
    main()
