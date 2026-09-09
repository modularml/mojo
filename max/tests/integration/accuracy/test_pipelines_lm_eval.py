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
"""Test for running pipelines_lm_eval."""

import json
import logging
import signal
import sys
from pathlib import Path

import pipelines_lm_eval
from click.testing import CliRunner
from max.pipelines.lib import generate_local_model_path

REPO_ID = "HuggingFaceTB/SmolLM-135M"

logger = logging.getLogger("max.pipelines")


def test_pipelines_lm_eval_smollm(tmp_path: Path) -> None:
    runner = CliRunner()
    output_dir = tmp_path / "lm-eval-output"

    try:
        local_model_path = generate_local_model_path(REPO_ID)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {REPO_ID} as config to PipelineConfig"
        )
        local_model_path = REPO_ID

    result = runner.invoke(
        pipelines_lm_eval.main,
        [
            "--pipelines-probe-port=8000",
            "--pipelines-probe-timeout=240",
            "--pipelines-arg=serve",
            f"--pipelines-arg=--model-path={local_model_path}",
            "--pipelines-arg=--quantization-encoding=float32",
            "--pipelines-arg=--max-length=512",
            "--pipelines-arg=--device-memory-utilization=0.3",
            "--lm-eval-arg=--model=local-completions",
            "--lm-eval-arg=--tasks=smol_task",
            f"--lm-eval-arg=--model_args=model={local_model_path},base_url=http://localhost:8000/v1/completions,tokenized_requests=False,num_concurrent=20,max_retries=3,max_length=512",
            f"--lm-eval-arg=--output_path={output_dir}",
            "--lm-eval-arg=--log_samples",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    results_dir = next(output_dir.iterdir())

    # Print predictions in case the test fails.
    samples_file = next(results_dir.glob("samples_*"))
    for n, line in enumerate(samples_file.read_text().split("\n")):
        line = line.strip()
        if not line:
            continue
        samples = json.loads(line)
        prompt = samples["doc"]["prompt"]
        response = samples["filtered_resps"][0]
        print(f"Prompt {n}: {prompt}")
        print(f"Response {n}: {response}")

    # Confirm that 3 out of the 4 results match (in test_prompts.json, one of
    # the "expected" predictions is incorrect).
    results_file = next(results_dir.glob("results_*"))
    results = json.loads(results_file.read_text().strip())
    assert results["results"]["smol_task"]["results_match,none"] == 0.75


def test_pipeline_sitter_is_alive() -> None:
    """Test that is_alive() correctly detects process death."""
    sitter = pipelines_lm_eval.PipelineSitter(
        [sys.executable, "-c", "import time; time.sleep(3600)"]
    )
    with sitter:
        assert sitter.is_alive()
        assert sitter._proc is not None
        # Kill the managed process
        sitter._proc.send_signal(signal.SIGKILL)
        sitter._proc.wait()
        assert not sitter.is_alive()


def test_server_crash_terminates_evaluator() -> None:
    """Test that a server crash during evaluation causes early exit."""
    sitter = pipelines_lm_eval.PipelineSitter(
        [sys.executable, "-c", "import time; time.sleep(3600)"]
    )
    with sitter:
        assert sitter._proc is not None
        # Kill the server to simulate a crash
        sitter._proc.send_signal(signal.SIGKILL)
        sitter._proc.wait()

        rc = pipelines_lm_eval.run_evaluator_with_crash_detection(
            [sys.executable, "-c", "import time; time.sleep(3600)"],
            sitter,
            poll_interval=0.1,
        )
        assert rc == 1


def test_health_probe_detects_hung_server() -> None:
    """Test that health probe failures trigger early exit."""
    # Start a process that stays alive but doesn't listen on any port.
    sitter = pipelines_lm_eval.PipelineSitter(
        [sys.executable, "-c", "import time; time.sleep(3600)"]
    )
    with sitter:
        rc = pipelines_lm_eval.run_evaluator_with_crash_detection(
            [sys.executable, "-c", "import time; time.sleep(3600)"],
            sitter,
            poll_interval=0.1,
            # Point at an unused port so every probe fails.
            health_probe_url="http://127.0.0.1:19876/health",
            health_probe_timeout=1.0,
            health_probe_max_consecutive_failures=3,
        )
        assert rc == 1


def test_pipeline_sitter_passes_env() -> None:
    """Test that PipelineSitter forwards env vars to the subprocess."""
    sitter = pipelines_lm_eval.PipelineSitter(
        [
            sys.executable,
            "-c",
            "import os, sys; sys.exit(0 if os.environ.get('TEST_HEALTH_VAR') == 'hello' else 1)",
        ],
        extra_env={"TEST_HEALTH_VAR": "hello"},
    )
    with sitter:
        assert sitter._proc is not None
        rc = sitter._proc.wait()
        assert rc == 0
