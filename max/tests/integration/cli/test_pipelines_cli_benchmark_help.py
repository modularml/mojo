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


from click.testing import CliRunner
from max._entrypoints import pipelines


def test_benchmark_subcommand_help() -> None:
    """Test that the benchmark help message works."""
    runner = CliRunner()
    result = runner.invoke(pipelines.main, ["benchmark", "--help"])
    assert result.exit_code == 0

    # Check if some benchmark specific options are present.
    assert "--dataset-name" in result.output
    assert "--dataset-path" in result.output
    assert "--num-prompts" in result.output
    assert "--seed" in result.output
