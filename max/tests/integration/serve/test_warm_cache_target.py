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
"""Test warm-cache command with --target flag for CUDA and HIP targets without requiring GPU."""

import subprocess

import python.runfiles

# Use SmolLM-135M for fast testing
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"


def _get_pipelines_binary() -> str:
    """Get the path to the pipelines binary from runfiles."""
    runfiles = python.runfiles.Create()
    assert runfiles is not None, "Unable to find runfiles tree"
    loc = runfiles.Rlocation("_main/max/python/max/_entrypoints/pipelines")
    assert loc is not None, "Unable to find pipelines entrypoint"
    return loc


def test_warm_cache_target_cuda() -> None:
    """Test warm-cache command with --target cuda flag.

    This test verifies that:
    - The pipelines warm-cache command with --target cuda:sm_80 succeeds
    - Virtual device mode is enabled
    - Model compiles successfully without requiring physical GPU
    - Process exits cleanly with exit code 0
    """
    pipelines_binary = _get_pipelines_binary()
    result = subprocess.run(
        [
            pipelines_binary,
            "warm-cache",
            "--model",
            MODEL_NAME,
            "--devices",
            "gpu:0",
            "--target",
            "cuda:sm_80",
        ],
        capture_output=True,
        text=True,
        timeout=180,  # 3 minutes should be plenty for compilation
    )

    # Verify successful exit
    assert result.returncode == 0, (
        f"Command failed with stderr:\n{result.stderr}"
    )

    # Verify virtual device mode was enabled
    assert (
        "Compiling for target: cuda (sm_80) using virtual devices"
        in result.stderr
    ), "Target compilation message not found"

    # Verify compilation succeeded (warm-cache doesn't print completion message, just exits)
    # Check that we got past initialization
    assert "Building, compiling, and initializing model" in result.stderr, (
        "Model compilation did not start"
    )


def test_warm_cache_target_cuda_default() -> None:
    """Test warm-cache command with --target cuda (default architecture).

    This test verifies that:
    - The pipelines warm-cache command with --target cuda succeeds
    - Default architecture (sm_80) is used
    - Virtual device mode is enabled
    - Model compiles successfully without requiring physical GPU
    """
    pipelines_binary = _get_pipelines_binary()
    result = subprocess.run(
        [
            pipelines_binary,
            "warm-cache",
            "--model",
            MODEL_NAME,
            "--devices",
            "gpu:0",
            "--target",
            "cuda",
        ],
        capture_output=True,
        text=True,
        timeout=180,  # 3 minutes should be plenty for compilation
    )

    # Verify successful exit
    assert result.returncode == 0, (
        f"Command failed with stderr:\n{result.stderr}"
    )

    # Verify default architecture was used
    assert (
        "Compiling for target: cuda (sm_80) using virtual devices"
        in result.stderr
    ), "Default sm_80 architecture not used"

    # Verify compilation succeeded
    assert "Building, compiling, and initializing model" in result.stderr, (
        "Model compilation did not start"
    )


def test_warm_cache_target_hip() -> None:
    """Test warm-cache command with --target hip flag.

    This test verifies that:
    - The pipelines warm-cache command with --target hip succeeds
    - Virtual device mode is enabled
    - Model compiles successfully without requiring physical GPU
    - Process exits cleanly with exit code 0
    """
    pipelines_binary = _get_pipelines_binary()
    result = subprocess.run(
        [
            pipelines_binary,
            "warm-cache",
            "--model",
            MODEL_NAME,
            "--devices",
            "gpu:0",
            "--target",
            "hip",
        ],
        capture_output=True,
        text=True,
        timeout=180,  # 3 minutes should be plenty for compilation
    )

    # Verify successful exit
    assert result.returncode == 0, (
        f"Command failed with stderr:\n{result.stderr}"
    )

    # Verify virtual device mode was enabled
    assert "Compiling for target: hip" in result.stderr, (
        "Target compilation message not found"
    )

    # Verify compilation succeeded
    assert "Building, compiling, and initializing model" in result.stderr, (
        "Model compilation did not start"
    )


def test_warm_cache_target_multiple_gpus() -> None:
    """Test warm-cache command with multiple virtual GPUs.

    This test verifies that:
    - The pipelines warm-cache command works with multiple device specs
    - Virtual device mode creates the correct number of virtual devices
    - Model compiles successfully without requiring physical GPUs

    Note: SmolLM-135M has 3 KV heads, so we use 3 GPUs to ensure divisibility.
    """
    pipelines_binary = _get_pipelines_binary()
    result = subprocess.run(
        [
            pipelines_binary,
            "warm-cache",
            "--model",
            MODEL_NAME,
            "--devices",
            "gpu:0,1,2",
            "--target",
            "cuda:sm_80",
        ],
        capture_output=True,
        text=True,
        timeout=180,  # 3 minutes should be plenty for compilation
    )

    # Verify successful exit
    assert result.returncode == 0, (
        f"Command failed with stderr:\n{result.stderr}"
    )

    # Verify virtual device mode was enabled with correct count
    # The system should create 3 virtual devices for gpu:0,1,2
    assert (
        "Compiling for target: cuda (sm_80) using virtual devices"
        in result.stderr
    ), "Target compilation message not found"

    # Verify compilation succeeded
    assert "Building, compiling, and initializing model" in result.stderr, (
        "Model compilation did not start"
    )
