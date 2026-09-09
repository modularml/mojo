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

import json

import pytest
from max._entrypoints import pipelines


def test_pipelines_list_json(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        pipelines.main(["list", "--json"])
    captured = capsys.readouterr()

    # Parse the JSON output
    output = json.loads(captured.out)

    # Verify the schema structure
    assert "architectures" in output
    assert isinstance(output["architectures"], dict)

    # Check structure of an architecture entry
    for arch_name, arch_data in output["architectures"].items():
        assert isinstance(arch_name, str)
        assert isinstance(arch_data, dict)
        assert "example_repo_ids" in arch_data
        assert isinstance(arch_data["example_repo_ids"], list)
        assert "supported_encodings" in arch_data
        assert isinstance(arch_data["supported_encodings"], list)

        # Check encoding entries
        for encoding in arch_data["supported_encodings"]:
            assert isinstance(encoding, str)
