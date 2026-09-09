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

"""Integration tests for ``max.benchmark.sweep_benchmark_serving``.

Percentile validation and the uploader protocol are covered by
``test_sweep_benchmark_serving_result_utils.py``; schema-driven CSV column
derivation is covered by ``test_model_csv.py``.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_mock
import yaml
from max.benchmark import sweep_benchmark_serving
from max.benchmark.benchmark_shared.metrics import (
    BenchmarkResult,
    TextGenAggregates,
)

pytestmark = pytest.mark.usefixtures("offline_dryrun_mocks")


@pytest.fixture
def workload_config(tmp_path: Path) -> Path:
    """A minimal workload YAML written to a temp path."""
    path = tmp_path / "workload.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset-name": "random",
                "random-input-len": 2000,
                "random-output-len": 100,
                "random-sys-prompt-ratio": 0.1,
            }
        )
    )
    return path


@pytest.fixture
def cmd_args(tmp_path: Path, workload_config: Path) -> list[str]:
    """CLI args shared across the multi-concurrency dry-run tests."""
    return [
        "--model",
        "HuggingFaceTB/SmolLM2-135M",
        "--workload-config",
        str(workload_config),
        "--max-concurrency",
        "1,2",
        "--num-prompts",
        "10",
        "--request-rate",
        "10",
        "--log-dir",
        str(tmp_path / "sweep-test-logs"),
        "--dry-run",
    ]


_NUM_PROMPTS_DURATION_WARNING = (
    "Neither --num-prompts nor --max-benchmark-duration-s is specified"
)


def test_missing_workload_config_is_allowed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A sweep without --workload-config is allowed (dataset/limits come from CLI)."""
    base_cmd_args = [
        "--model",
        "HuggingFaceTB/SmolLM2-135M",
        "--max-concurrency",
        "1",
        "--num-prompts",
        "10",
        "--dry-run",
    ]

    sweep_benchmark_serving.main(base_cmd_args)
    stdout, _stderr = capsys.readouterr()
    assert "Dry run:" in stdout


def test_warn_missing_workload_config_with_upload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """--upload-results without --workload-config logs a warning but proceeds."""
    base_cmd_args = [
        "--model",
        "HuggingFaceTB/SmolLM2-135M",
        "--max-concurrency",
        "1",
        "--num-prompts",
        "10",
        "--upload-results",
        "--dry-run",
    ]

    with caplog.at_level(logging.WARNING, logger="sweep-benchmark-serving"):
        sweep_benchmark_serving.main(base_cmd_args)

    expected_fragment = (
        "--workload-config is not set while --upload-results is set"
    )
    assert any(
        expected_fragment in record.message for record in caplog.records
    ), (
        f"Expected warning containing {expected_fragment!r} in log records:\n"
        f"{[r.message for r in caplog.records]}"
    )


def test_correct_number_of_runs(
    cmd_args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Verifies the correct number of benchmark commands are generated."""
    sweep_benchmark_serving.main(cmd_args)
    stdout, _stderr = capsys.readouterr()

    actual_runs = stdout.count("Dry run:")
    expected_runs = 2
    assert actual_runs == expected_runs, (
        f"Expected {expected_runs} 'Dry run:' counts, found {actual_runs}\nOutput:\n{stdout}"
    )


def test_warns_if_num_prompts_and_max_benchmark_duration_s_not_provided(
    tmp_path: Path,
    workload_config: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Emits a warning when num-prompts and max-benchmark-duration-s are not provided."""
    cmd_missing_args = [
        "--model",
        "HuggingFaceTB/SmolLM2-135M",
        "--workload-config",
        str(workload_config),
        "--max-concurrency",
        "1",
        "--log-dir",
        str(tmp_path),
        "--dry-run",
    ]

    with caplog.at_level(logging.WARNING):
        sweep_benchmark_serving.main(cmd_missing_args)

    assert any(
        _NUM_PROMPTS_DURATION_WARNING in record.message
        for record in caplog.records
    ), (
        f"Expected defaulting warning in log records:\n{[r.message for r in caplog.records]}"
    )


def test_override_num_prompts_if_set_explicitly(
    tmp_path: Path,
    workload_config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When --num-prompts is set, no defaulting warning and the value is used."""
    cmd_missing_args = [
        "--model",
        "HuggingFaceTB/SmolLM2-135M",
        "--workload-config",
        str(workload_config),
        "--num-prompts",
        "700",
        "--max-concurrency",
        "1",
        "--log-dir",
        str(tmp_path),
        "--dry-run",
    ]

    sweep_benchmark_serving.main(cmd_missing_args)
    stdout, stderr = capsys.readouterr()

    assert _NUM_PROMPTS_DURATION_WARNING not in stderr, (
        f"Unexpected defaulting warning in stderr:\n{stderr}"
    )
    assert "num_prompts=700" in stdout


def test_override_benchmark_duration_s(
    tmp_path: Path,
    workload_config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When --max-benchmark-duration-s is set, no defaulting warning."""
    cmd_missing_args = [
        "--model",
        "HuggingFaceTB/SmolLM2-135M",
        "--workload-config",
        str(workload_config),
        "--max-benchmark-duration-s",
        "10",
        "--max-concurrency",
        "1",
        "--log-dir",
        str(tmp_path),
        "--dry-run",
    ]

    sweep_benchmark_serving.main(cmd_missing_args)
    _stdout, stderr = capsys.readouterr()

    assert _NUM_PROMPTS_DURATION_WARNING not in stderr, (
        f"Unexpected defaulting warning in stderr:\n{stderr}"
    )


def test_default_arguments(
    tmp_path: Path,
    workload_config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default host, port, and endpoint appear in dry-run output."""
    minimal_cmd_args = [
        "--model",
        "HuggingFaceTB/SmolLM2-135M",
        "--workload-config",
        str(workload_config),
        "--max-concurrency",
        "1",
        "--num-prompts",
        "1",
        "--log-dir",
        str(tmp_path),
        "--dry-run",
    ]

    sweep_benchmark_serving.main(minimal_cmd_args)
    stdout, _stderr = capsys.readouterr()

    dry_run_lines = [
        line for line in stdout.strip().split("\n") if "Dry run:" in line
    ]
    assert len(dry_run_lines) == 1, (
        f"Expected one dry-run line, got: {dry_run_lines}"
    )
    dry_run_output = dry_run_lines[0]

    assert "host=localhost" in dry_run_output, (
        f"host=localhost not found in: {dry_run_output}"
    )
    assert "port=8000" in dry_run_output, (
        f"port=8000 not found in: {dry_run_output}"
    )
    assert "endpoint=/v1/chat/completions" in dry_run_output, (
        f"endpoint=/v1/chat/completions not found in: {dry_run_output}"
    )


def test_upload_results(
    cmd_args: list[str],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With --upload-results, a warning is logged without cluster info.

    Benchmark ``--dry-run`` yields no ``result_filename``, so the result
    writer skips the upload callback (no extra ``Dry run:`` lines).
    """
    cmd_upload_results_args = cmd_args + ["--upload-results"]

    with caplog.at_level(logging.WARNING):
        sweep_benchmark_serving.main(cmd_upload_results_args)
    stdout, _stderr = capsys.readouterr()

    actual_dry_run = stdout.count("Dry run:")
    # Two benchmark dry-runs only (mc=1 and mc=2); upload is skipped without JSON path.
    assert actual_dry_run == 2, (
        f'Expected 2 benchmark "Dry run:" lines, got {actual_dry_run}:\n{stdout}'
    )
    assert "Uploading benchmark results" not in stdout

    expected_warning = "Warning: uploading results without cluster information"
    assert any(
        expected_warning in record.message for record in caplog.records
    ), (
        f"Expected warning in log records, got: {[r.message for r in caplog.records]}"
    )


def test_latency_percentiles_invalid_format(
    cmd_args: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid percentile tokens surface as a subprocess error."""
    cmd_invalid_args = cmd_args + [
        "--latency-percentiles",
        "invalid,not_a_number",
    ]

    with pytest.raises(ValueError, match="invalid literal for int\\(\\)"):
        sweep_benchmark_serving.main(cmd_invalid_args)
    _ = capsys.readouterr()


def test_latency_percentiles_help_message(
    cmd_args: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--help`` mentions ``--latency-percentiles``."""
    help_cmd_args = cmd_args + ["--help"]

    sweep_benchmark_serving.main(help_cmd_args)
    stdout, _stderr = capsys.readouterr()

    assert "--latency-percentiles" in stdout, (
        f"--latency-percentiles not found in help:\n{stdout}"
    )


def test_cli_max_benchmark_duration_s_overrides_workload_yaml(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI --max-benchmark-duration-s wins over workload YAML."""
    yaml_content = """dataset-name: random
random-input-len: 2000
random-output-len: 100
random-sys-prompt-ratio: 0.1
max-benchmark-duration-s: 1500
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as temp_file:
        temp_file.write(yaml_content)
        temp_file.flush()

        cmd_test_args = [
            "--model",
            "HuggingFaceTB/SmolLM2-135M",
            "--workload-config",
            temp_file.name,
            "--max-benchmark-duration-s",
            "300",
            "--max-concurrency",
            "1",
            "--log-dir",
            str(tmp_path),
            "--dry-run",
        ]

        sweep_benchmark_serving.main(cmd_test_args)
    stdout, _stderr = capsys.readouterr()

    assert "max_benchmark_duration_s=300" in stdout, stdout
    assert "max_benchmark_duration_s=1500" not in stdout, stdout


def test_cli_max_concurrency_overrides_workload_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--max-concurrency`` on the CLI overrides ``max-concurrency`` in workload YAML."""
    yaml_content = """dataset-name: random
random-input-len: 100
random-output-len: 50
max-concurrency: 100
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as temp_file:
        temp_file.write(yaml_content)
        temp_file.flush()

        cmd_test_args = [
            "--model",
            "HuggingFaceTB/SmolLM2-135M",
            "--workload-config",
            temp_file.name,
            "--max-concurrency",
            "5",
            "--num-prompts",
            "10",
            "--log-dir",
            str(tmp_path),
            "--dry-run",
        ]

        sweep_benchmark_serving.main(cmd_test_args)
        stdout, _stderr = capsys.readouterr()

        assert "max_concurrency=100" not in stdout, (
            f"Workload max-concurrency should be overridden by CLI:\n{stdout}"
        )

        assert "max_concurrency=5" in stdout, (
            f"CLI max-concurrency not found in output:\n{stdout}"
        )


@pytest.mark.parametrize(
    ("status_code", "response_text"),
    [
        (400, "Prefix caching is not enabled"),
        (404, "404 page not found"),
    ],
)
def test_flush_prefix_cache_soft_status_logs_warning_not_raises(
    caplog: pytest.LogCaptureFixture,
    status_code: int,
    response_text: str,
) -> None:
    """HTTP 400 (prefix caching disabled) and 404 (route not exposed) should warn, not raise."""
    from max.benchmark.benchmark_serving import flush_prefix_cache

    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.text = response_text

    with patch("requests.post", return_value=mock_response):
        flush_prefix_cache("modular", "localhost", 8000, dry_run=False)

    assert any(
        "skipping cache flush" in record.message for record in caplog.records
    ), f"Expected warning in logs, got: {[r.message for r in caplog.records]}"


def test_flush_prefix_cache_other_errors_raise() -> None:
    """HTTP 500 should still raise RuntimeError."""
    from max.benchmark.benchmark_serving import flush_prefix_cache

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Failed to flush prefix cache"):
            flush_prefix_cache("modular", "localhost", 8000, dry_run=False)


def test_flush_prefix_cache_proxy_wrapped_404_logs_warning_not_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mammoth proxy wraps engine 404s in 502; treat unanimous 404 children as skip."""
    from max.benchmark.benchmark_serving import flush_prefix_cache

    proxy_body = {
        "status": "error",
        "results": [
            {
                "endpoint": "http://10.42.138.154:8000",
                "name": "engine-0",
                "statusCode": 404,
                "success": False,
                "error": '{"detail":"Not Found"}',
            }
        ],
        "error": "failed to reset prefix cache for one or more endpoints",
    }
    mock_response = MagicMock()
    mock_response.status_code = 502
    mock_response.content = b"{...}"
    mock_response.json.return_value = proxy_body
    mock_response.text = str(proxy_body)

    with patch("requests.post", return_value=mock_response):
        flush_prefix_cache("vllm-chat", "localhost", 8000, dry_run=False)

    assert any(
        "skipping cache flush" in record.message for record in caplog.records
    ), f"Expected warning in logs, got: {[r.message for r in caplog.records]}"


def test_flush_prefix_cache_proxy_wrapped_mixed_statuses_raises() -> None:
    """Proxy 502 with mixed children (not all 404) should still raise."""
    from max.benchmark.benchmark_serving import flush_prefix_cache

    proxy_body = {
        "status": "error",
        "results": [
            {"statusCode": 404, "success": False},
            {"statusCode": 500, "success": False},
        ],
    }
    mock_response = MagicMock()
    mock_response.status_code = 502
    mock_response.content = b"{...}"
    mock_response.json.return_value = proxy_body
    mock_response.text = str(proxy_body)

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Failed to flush prefix cache"):
            flush_prefix_cache("vllm-chat", "localhost", 8000, dry_run=False)


def test_flush_prefix_cache_base_url_uses_url_and_bearer_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--base-url targets the remote endpoint and carries bearer auth."""
    from max.benchmark.benchmark_serving import flush_prefix_cache

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("requests.post", return_value=mock_response) as post:
        flush_prefix_cache(
            "modular",
            "localhost",
            8000,
            dry_run=False,
            base_url="https://example.com/",
        )

    post.assert_called_once_with(
        "https://example.com/reset_prefix_cache",
        headers={"Authorization": "Bearer test-key"},
    )


def test_flush_prefix_cache_no_base_url_uses_host_port_without_auth() -> None:
    """Local servers keep the http://host:port URL and send no auth."""
    from max.benchmark.benchmark_serving import flush_prefix_cache

    mock_response = MagicMock()
    mock_response.status_code = 200

    with patch("requests.post", return_value=mock_response) as post:
        flush_prefix_cache("modular", "localhost", 8000, dry_run=False)

    post.assert_called_once_with(
        "http://localhost:8000/reset_prefix_cache", headers=None
    )


# ===========================================================================
# result_filename plumbing tests
# ===========================================================================


def test_result_filename_reaches_main_with_parsed_args(
    tmp_path: Path,
    workload_config: Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """config.result_filename must not be cleared before main_with_parsed_args is called.

    Regression guard: if run_sweep reintroduces ``config.result_filename = None``
    before calling main_with_parsed_args, the result JSON for non-sweep CI runs
    will silently never be written and the "submit results" step will fail with
    ``ls: cannot access 'results': No such file or directory``.
    """
    result_path = str(tmp_path / "result.json")
    received: dict[str, str | None] = {}

    def capture_config(config: object, **kwargs: object) -> list[object]:
        received["result_filename"] = getattr(
            config, "result_filename", "NOT_SET"
        )
        return []

    mocker.patch(
        "max.benchmark.sweep_benchmark_serving.benchmark_serving_main",
        side_effect=capture_config,
    )
    sweep_benchmark_serving.main(
        [
            "--model",
            "HuggingFaceTB/SmolLM2-135M",
            "--workload-config",
            str(workload_config),
            "--max-concurrency",
            "1",
            "--num-prompts",
            "10",
            "--result-filename",
            result_path,
            "--log-dir",
            str(tmp_path),
        ]
    )

    assert received.get("result_filename") == result_path, (
        f"config.result_filename was cleared before main_with_parsed_args. "
        f"Expected {result_path!r}, got {received.get('result_filename')!r}."
    )


def test_result_filename_none_when_not_provided(
    tmp_path: Path,
    workload_config: Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """When --result-filename is not passed, config.result_filename must be None."""
    received: dict[str, str | None] = {}

    def capture_config(config: object, **kwargs: object) -> list[object]:
        received["result_filename"] = getattr(
            config, "result_filename", "NOT_SET"
        )
        return []

    mocker.patch(
        "max.benchmark.sweep_benchmark_serving.benchmark_serving_main",
        side_effect=capture_config,
    )
    sweep_benchmark_serving.main(
        [
            "--model",
            "HuggingFaceTB/SmolLM2-135M",
            "--workload-config",
            str(workload_config),
            "--max-concurrency",
            "1",
            "--num-prompts",
            "10",
            "--log-dir",
            str(tmp_path),
        ]
    )

    assert received.get("result_filename") is None


# ===========================================================================
# JSON output tests
# ===========================================================================


def _real_text_result(
    *, completed: int = 5, duration: float = 1.0
) -> BenchmarkResult:
    """A minimal real result: ``save_result_json`` reads ``result_groups``
    from the model field, so a ``MagicMock`` no longer suffices."""
    return BenchmarkResult(
        task_type="text",
        max_concurrency=1,
        text_data=TextGenAggregates(
            duration=duration,
            completed=completed,
            failures=0,
            request_throughput=completed / duration,
            total_input=10,
            total_output=20,
            nonempty_response_chunks=20,
            max_input=10,
            max_output=20,
            max_total=30,
            global_cached_token_rate=0.0,
        ),
    )


def test_save_result_json_writes_valid_json(
    tmp_path: Path, workload_config: Path
) -> None:
    """save_result_json must write a file containing the expected top-level keys."""
    from max.benchmark.benchmark_serving import save_result_json

    result_path = str(tmp_path / "result.json")
    config = sweep_benchmark_serving.parse_args(
        [
            "--model",
            "myorg/mymodel",
            "--workload-config",
            str(workload_config),
            "--result-filename",
            result_path,
            "--backend",
            "modular",
        ]
    )

    save_result_json(
        config.result_filename,
        config,
        _real_text_result(completed=5),
        benchmark_task="text-generation",
        model_id="myorg/mymodel",
        tokenizer_id="myorg/mymodel",
        request_rate=10.0,
        record_max_concurrency=config.max_concurrency[0],
    )

    assert Path(result_path).exists(), (
        f"Expected result JSON at {result_path!r} but no file was created."
    )
    with open(result_path) as f:
        data = json.load(f)

    assert data["backend"] == "modular"
    assert data["model_id"] == "myorg/mymodel"
    assert data["num_prompts"] == 5
    assert data["request_rate"] == 10.0
    assert "date" in data
    assert "duration" in data
    assert data["result_groups"]["summary"]["completed"] == 5


def test_result_json_written_at_specified_path(
    tmp_path: Path,
    workload_config: Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Exactly one JSON file must be written at the path passed via --result-filename.

    The mock simulates what main_with_parsed_args does: write the JSON when
    config.result_filename is set, then return an empty result list (no sweep).
    """
    result_path = tmp_path / "output" / "result.json"

    def fake_benchmark(config: object, **kwargs: object) -> list[object]:
        filename = getattr(config, "result_filename", None)
        if filename:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, "w") as f:
                json.dump(
                    {
                        "model_id": getattr(config, "model", None),
                        "backend": getattr(config, "backend", None),
                    },
                    f,
                )
        return []

    mocker.patch(
        "max.benchmark.sweep_benchmark_serving.benchmark_serving_main",
        side_effect=fake_benchmark,
    )
    sweep_benchmark_serving.main(
        [
            "--model",
            "myorg/mymodel",
            "--workload-config",
            str(workload_config),
            "--max-concurrency",
            "1",
            "--num-prompts",
            "10",
            "--result-filename",
            str(result_path),
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )

    assert result_path.exists(), (
        f"Expected result JSON at {result_path!r} but no file was created."
    )
    # Verify there is exactly one JSON in the output dir (no extra files).
    json_files = list(result_path.parent.glob("*.json"))
    assert len(json_files) == 1, (
        f"Expected exactly 1 JSON file, found {len(json_files)}: {json_files}"
    )
    with open(result_path) as f:
        data = json.load(f)
    assert data["model_id"] == "myorg/mymodel"
    assert data["backend"] == "modular"


# ===========================================================================
# CLI vs workload-config precedence tests
# ===========================================================================


def test_apply_workload_skips_explicitly_set_fields() -> None:
    """Fields in model_fields_set must not be overwritten by workload YAML."""
    from max.benchmark.benchmark_serving import _apply_workload_to_config
    from max.benchmark.benchmark_shared.config import ServingBenchmarkConfig

    config = ServingBenchmarkConfig(
        model="myorg/mymodel",
        request_rate=[5.0],
    )
    # Both `model` and `request_rate` are now in model_fields_set.
    assert "request_rate" in config.model_fields_set
    assert "model" in config.model_fields_set

    workload = {
        "request-rate": "999",
        "model": "other/model",
        "dataset-name": "random",
    }

    _apply_workload_to_config(config, workload)

    assert list(config.request_rate) == [5.0], (
        f"CLI request_rate should not be overwritten by workload YAML;"
        f" got {config.request_rate}"
    )
    assert config.model == "myorg/mymodel", (
        f"CLI model should not be overwritten by workload YAML;"
        f" got {config.model}"
    )
    # dataset_name was not explicitly set, so it should be applied from YAML.
    assert config.dataset_name == "random", (
        f"Workload dataset-name should be applied when not set by CLI;"
        f" got {config.dataset_name}"
    )


def test_apply_workload_sets_unset_fields() -> None:
    """Fields absent from model_fields_set must be filled from workload YAML."""
    from max.benchmark.benchmark_serving import _apply_workload_to_config
    from max.benchmark.benchmark_shared.config import ServingBenchmarkConfig

    config = ServingBenchmarkConfig(model="myorg/mymodel")
    assert "random_input_len" not in config.model_fields_set

    workload = {"random-input-len": "512", "random-output-len": "128"}
    _apply_workload_to_config(config, workload)

    assert config.random_input_len == "512", (
        f"Workload random-input-len should be applied; got {config.random_input_len}"
    )
    assert config.random_output_len == "128", (
        f"Workload random-output-len should be applied; got {config.random_output_len}"
    )


def test_cli_request_rate_overrides_workload_yaml(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CLI --request-rate wins over request-rate in workload YAML."""
    yaml_content = """dataset-name: random
random-input-len: 512
random-output-len: 128
request-rate: "999"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as temp_file:
        temp_file.write(yaml_content)
        temp_file.flush()

        cmd_test_args = [
            "--model",
            "HuggingFaceTB/SmolLM2-135M",
            "--workload-config",
            temp_file.name,
            "--request-rate",
            "7",
            "--num-prompts",
            "10",
            "--max-concurrency",
            "1",
            "--log-dir",
            str(tmp_path),
            "--dry-run",
        ]

        with caplog.at_level(logging.INFO):
            sweep_benchmark_serving.main(cmd_test_args)
    stdout, _stderr = capsys.readouterr()

    assert "request_rate=7" in stdout or "request_rate=7.0" in stdout, (
        f"Expected CLI request_rate=7 in output:\n{stdout}"
    )
    assert "request_rate=999" not in stdout, (
        f"Workload request_rate should not override CLI value:\n{stdout}"
    )
    # The log should record that CLI took precedence for request-rate.
    assert any(
        "CLI flag --request-rate takes precedence" in r.message
        for r in caplog.records
    ), (
        f"Expected CLI precedence log for request-rate;"
        f" got: {[r.message for r in caplog.records]}"
    )


def test_workload_yaml_applies_when_cli_not_set(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Workload YAML fields apply when not explicitly specified on the CLI."""
    yaml_content = """dataset-name: random
random-input-len: 256
random-output-len: 64
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as temp_file:
        temp_file.write(yaml_content)
        temp_file.flush()

        cmd_test_args = [
            "--model",
            "HuggingFaceTB/SmolLM2-135M",
            "--workload-config",
            temp_file.name,
            "--num-prompts",
            "10",
            "--max-concurrency",
            "1",
            "--log-dir",
            str(tmp_path),
            "--dry-run",
        ]

        with caplog.at_level(logging.INFO):
            sweep_benchmark_serving.main(cmd_test_args)

    # Both workload fields should be logged as "Applying workload YAML value".
    applied_messages = [
        r.message
        for r in caplog.records
        if "Applying workload YAML value" in r.message
    ]
    keys_applied = {
        msg.split("--")[1].split("=")[0] for msg in applied_messages
    }
    assert "random-input-len" in keys_applied, (
        f"Expected 'random-input-len' to be applied from workload YAML;"
        f" log messages: {applied_messages}"
    )
    assert "random-output-len" in keys_applied, (
        f"Expected 'random-output-len' to be applied from workload YAML;"
        f" log messages: {applied_messages}"
    )


def test_upload_path_writes_one_json_per_concurrency(
    tmp_path: Path,
    workload_config: Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Upload mode must call save_result_json once per concurrency level."""
    from max.benchmark.benchmark_serving import BenchmarkRunResult

    fake_results = [
        BenchmarkRunResult(
            max_concurrency=1,
            request_rate=float("inf"),
            num_prompts=10,
            result=MagicMock(),
        ),
        BenchmarkRunResult(
            max_concurrency=2,
            request_rate=float("inf"),
            num_prompts=10,
            result=MagicMock(),
        ),
    ]

    saved_filenames: list[str] = []

    def capture_save(
        result_filename: str | None, *args: object, **kwargs: object
    ) -> None:
        saved_filenames.append(result_filename or "")

    mocker.patch(
        "max.benchmark.sweep_benchmark_serving.benchmark_serving_main",
        return_value=fake_results,
    )
    mocker.patch(
        "max.benchmark.sweep_benchmark_serving.save_result_json",
        side_effect=capture_save,
    )
    mocker.patch("max.benchmark.sweep_benchmark_serving.CsvStreamWriter")

    sweep_benchmark_serving.main(
        [
            "--model",
            "myorg/mymodel",
            "--workload-config",
            str(workload_config),
            "--max-concurrency",
            "1,2",
            "--num-prompts",
            "10",
            "--upload-results",
            "--log-dir",
            str(tmp_path),
        ],
        uploader=MagicMock(),
    )

    assert len(saved_filenames) == 2, (
        f"Expected save_result_json called once per concurrency (2 times), "
        f"got {len(saved_filenames)}."
    )
    for filename, expected_mc in zip(saved_filenames, [1, 2], strict=False):
        assert f"results-{expected_mc}-median.json" in filename, (
            f"Expected results-{expected_mc}-median.json in filename, "
            f"got {filename!r}"
        )


def test_upload_writes_correct_data_to_correct_files(
    tmp_path: Path,
    workload_config: Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Each concurrency level's JSON file must contain that level's results.

    Regression test: a config-mutation bug caused the generator to resume
    with a stale result_filename on the second iteration, writing mc2 data
    into mc1's file and burying mc1's data in a .orig backup.

    The fake generator here mimics the real generator's behavior: it reads
    config.result_filename lazily on each resume.  If run_sweep mutates
    config between yields (the bug), the second resume picks up the first
    iteration's stale path and the file-content assertions fail.
    """
    from collections.abc import Iterator

    from max.benchmark.benchmark_serving import (
        BenchmarkRunResult,
        save_result_json,
    )
    from max.benchmark.benchmark_shared.config import ServingBenchmarkConfig

    def fake_benchmark_serving_main(
        config: ServingBenchmarkConfig,
        **kwargs: object,
    ) -> Iterator[BenchmarkRunResult]:
        assert config.model is not None
        for mc in [1, 2]:
            # ``duration == float(mc)`` is the per-iteration sentinel: stale
            # mc1 data in mc2's file (or vice versa) shows as the wrong value.
            result = _real_text_result(completed=5, duration=float(mc))
            save_result_json(
                config.result_filename,
                config,
                result,
                benchmark_task="text-generation",
                model_id=config.model,
                tokenizer_id=config.model,
                request_rate=float(mc),
                record_max_concurrency=mc,
            )
            yield BenchmarkRunResult(
                max_concurrency=mc,
                request_rate=float(mc),
                num_prompts=10,
                result=result,
            )

    mocker.patch(
        "max.benchmark.sweep_benchmark_serving.benchmark_serving_main",
        side_effect=fake_benchmark_serving_main,
    )
    mocker.patch("max.benchmark.sweep_benchmark_serving.CsvStreamWriter")

    sweep_benchmark_serving.main(
        [
            "--model",
            "myorg/mymodel",
            "--workload-config",
            str(workload_config),
            "--max-concurrency",
            "1,2",
            "--num-prompts",
            "10",
            "--upload-results",
            "--backend",
            "modular",
            "--log-dir",
            str(tmp_path),
        ],
        uploader=MagicMock(),
    )

    mc1_file = tmp_path / "results-1-median.json"
    mc2_file = tmp_path / "results-2-median.json"

    assert mc1_file.exists(), "results-1-median.json was not written"
    assert mc2_file.exists(), "results-2-median.json was not written"
    assert not (tmp_path / "results-1-median.json.orig").exists(), (
        "results-1-median.json.orig found — the generator wrote stale data "
        "into mc1's file, meaning config.result_filename was mutated between iterations"
    )

    mc1_data = json.loads(mc1_file.read_text())
    mc2_data = json.loads(mc2_file.read_text())
    assert mc1_data.get("duration") == 1.0, (
        f"results-1-median.json contains mc2 data (duration={mc1_data.get('duration')!r})"
    )
    assert mc2_data.get("duration") == 2.0, (
        f"results-2-median.json contains wrong data (duration={mc2_data.get('duration')!r})"
    )


def test_upload_path_single_run_no_max_concurrency(
    tmp_path: Path,
    workload_config: Path,
    mocker: pytest_mock.MockerFixture,
) -> None:
    """run_sweep must call save_result_json for a non-sweep (single-run) upload.

    When upload_results=True and max_concurrency is not set, run_sweep must
    still write a JSON — the result_dict/metrics guard must not silently skip
    the upload for single-run jobs migrated from the old out-of-process path.
    """
    from max.benchmark.benchmark_serving import BenchmarkRunResult

    fake_results = [
        BenchmarkRunResult(
            max_concurrency=None,
            request_rate=float("inf"),
            num_prompts=10,
            result=MagicMock(),
        )
    ]

    saved_filenames: list[str] = []

    def capture_save(
        result_filename: str | None, *args: object, **kwargs: object
    ) -> None:
        saved_filenames.append(result_filename or "")

    mocker.patch(
        "max.benchmark.sweep_benchmark_serving.benchmark_serving_main",
        return_value=fake_results,
    )
    mocker.patch(
        "max.benchmark.sweep_benchmark_serving.save_result_json",
        side_effect=capture_save,
    )
    mocker.patch("max.benchmark.sweep_benchmark_serving.CsvStreamWriter")

    sweep_benchmark_serving.main(
        [
            "--model",
            "myorg/mymodel",
            "--workload-config",
            str(workload_config),
            "--num-prompts",
            "10",
            "--upload-results",
            "--log-dir",
            str(tmp_path),
        ],
        uploader=MagicMock(),
    )

    assert len(saved_filenames) == 1, (
        f"Expected save_result_json called once for single-run upload, "
        f"got {len(saved_filenames)}."
    )
    assert "results-None-median.json" in saved_filenames[0], (
        f"Expected results-None-median.json in filename, "
        f"got {saved_filenames[0]!r}."
    )
