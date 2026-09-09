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

from __future__ import annotations

import logging
import resource
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from max.benchmark.benchmark_shared import utils as benchmark_utils
from max.benchmark.benchmark_shared.utils import (
    argmedian,
    exceeds_deadline,
    get_tokenizer,
    int_or_none,
    is_castable_to_int,
    parse_comma_separated,
    print_section,
    set_ulimit,
    wait_for_server_ready,
)
from pytest_mock import MockerFixture


def test_get_tokenizer_passes_model_max_length_when_provided(
    mocker: MockerFixture,
) -> None:
    from_pretrained = mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
    )
    mocker.patch(
        "transformers.AutoConfig.from_pretrained",
        return_value=MagicMock(architectures=[]),
    )

    tokenizer = get_tokenizer(
        "repo/model",
        revision="abc123",
        model_max_length=4096,
        trust_remote_code=True,
    )

    assert tokenizer is from_pretrained.return_value
    assert tokenizer._resolved_revision == "abc123"
    from_pretrained.assert_called_once_with(
        "repo/model",
        model_max_length=4096,
        trust_remote_code=True,
        revision="abc123",
        local_files_only=False,
    )


def test_get_tokenizer_omits_model_max_length_when_unspecified(
    mocker: MockerFixture,
) -> None:
    from_pretrained = mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
    )
    mocker.patch(
        "transformers.AutoConfig.from_pretrained",
        return_value=MagicMock(architectures=[]),
    )

    tokenizer = get_tokenizer(
        "repo/model", revision=None, trust_remote_code=False
    )

    assert tokenizer is from_pretrained.return_value
    assert tokenizer._resolved_revision is None
    from_pretrained.assert_called_once_with(
        "repo/model",
        trust_remote_code=False,
        revision=None,
        local_files_only=False,
    )


def test_get_tokenizer_forwards_local_files_only(
    mocker: MockerFixture,
) -> None:
    """A cache-only load keeps the Hub out of both transformers lookups."""
    from_pretrained = mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
    )
    autoconfig = mocker.patch(
        "transformers.AutoConfig.from_pretrained",
        return_value=MagicMock(architectures=[]),
    )

    get_tokenizer(
        "repo/model",
        revision="abc123",
        trust_remote_code=False,
        local_files_only=True,
    )

    from_pretrained.assert_called_once_with(
        "repo/model",
        trust_remote_code=False,
        revision="abc123",
        local_files_only=True,
    )
    autoconfig.assert_called_once_with(
        "repo/model",
        trust_remote_code=False,
        revision="abc123",
        local_files_only=True,
    )


def test_get_tokenizer_stashes_resolved_architectures(
    mocker: MockerFixture,
) -> None:
    mocker.patch("transformers.AutoTokenizer.from_pretrained")
    mocker.patch(
        "transformers.AutoConfig.from_pretrained",
        return_value=MagicMock(architectures=["LlamaForCausalLM"]),
    )

    tokenizer = get_tokenizer("repo/model", revision=None)

    assert tokenizer._resolved_architectures == ["LlamaForCausalLM"]


def test_get_tokenizer_skips_autoconfig_when_architectures_provided(
    mocker: MockerFixture,
) -> None:
    """A precomputed architecture list bypasses the AutoConfig Hub lookup."""
    mocker.patch("transformers.AutoTokenizer.from_pretrained")
    autoconfig = mocker.patch("transformers.AutoConfig.from_pretrained")

    tokenizer = get_tokenizer(
        "repo/model", revision=None, architectures=["LlamaForCausalLM"]
    )

    autoconfig.assert_not_called()
    assert tokenizer._resolved_architectures == ["LlamaForCausalLM"]


def test_get_tokenizer_applies_kimi_override_from_provided_architectures(
    mocker: MockerFixture,
) -> None:
    """The Kimi encode override works from a forwarded architecture list."""
    inner = MagicMock()
    real_tokenizer = MagicMock()
    real_tokenizer.encode = inner
    mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=real_tokenizer,
    )
    mocker.patch("transformers.AutoConfig.from_pretrained")

    tokenizer = get_tokenizer(
        "repo/model",
        revision=None,
        architectures=["KimiK25ForConditionalGeneration"],
    )
    tokenizer.encode("hello", add_special_tokens=True)

    _, kwargs = inner.call_args
    assert "add_special_tokens" not in kwargs
    assert kwargs["allow_special_tokens"] is True


def test_get_tokenizer_warns_once_on_autoconfig_failure(
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing AutoConfig probe warns once per model path, not per load."""
    mocker.patch("transformers.AutoTokenizer.from_pretrained")
    mocker.patch(
        "transformers.AutoConfig.from_pretrained",
        side_effect=OSError("cannot reach hub"),
    )
    benchmark_utils._reported_autoconfig_failures.discard("repo/unreachable")

    with caplog.at_level(
        logging.WARNING, logger="max.benchmark.benchmark_shared.utils"
    ):
        first = get_tokenizer("repo/unreachable", revision=None)
        second = get_tokenizer("repo/unreachable", revision=None)

    assert first._resolved_architectures == []
    assert second._resolved_architectures == []
    failures = [
        r
        for r in caplog.records
        if "AutoConfig.from_pretrained failed" in r.getMessage()
    ]
    assert len(failures) == 1


def test_quiet_hf_hub_retries_lowers_then_restores_level() -> None:
    hub_logger = logging.getLogger("huggingface_hub")
    hub_logger.setLevel(logging.WARNING)

    with benchmark_utils._quiet_hf_hub_retries():
        assert hub_logger.level == logging.ERROR

    assert hub_logger.level == logging.WARNING


def test_set_ulimit_updates_soft_limit_when_needed(
    mocker: MockerFixture,
) -> None:
    getrlimit = mocker.patch("resource.getrlimit", return_value=(1024, 65535))
    setrlimit = mocker.patch("resource.setrlimit")

    set_ulimit(target_soft_limit=4096)

    getrlimit.assert_called_once_with(resource.RLIMIT_NOFILE)
    setrlimit.assert_called_once_with(
        resource.RLIMIT_NOFILE,
        (4096, 65535),
    )


def test_set_ulimit_skips_update_when_soft_limit_is_high_enough(
    mocker: MockerFixture,
) -> None:
    mocker.patch("resource.getrlimit", return_value=(8192, 65535))
    setrlimit = mocker.patch("resource.setrlimit")

    set_ulimit(target_soft_limit=4096)

    setrlimit.assert_not_called()


def test_print_section_formats_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_section(" Shared Utilities ", char="=")

    expected = "{s:{c}^{n}}\n".format(
        s=" Shared Utilities ",
        n=50,
        c="=",
    )
    assert capsys.readouterr().out == expected


# ---- parse_comma_separated ----


def test_parse_comma_separated_single_value() -> None:
    assert parse_comma_separated("42", int) == [42]


def test_parse_comma_separated_multiple_values() -> None:
    assert parse_comma_separated("1,2,4", int) == [1, 2, 4]


def test_parse_comma_separated_strips_whitespace() -> None:
    assert parse_comma_separated(" 10 , 20 , 30 ", float) == [10.0, 20.0, 30.0]


def test_parse_comma_separated_none_returns_default() -> None:
    assert parse_comma_separated(None, int) == [None]


def test_parse_comma_separated_none_with_explicit_default() -> None:
    assert parse_comma_separated(None, int, default=0) == [0]


def test_parse_comma_separated_with_int_or_none() -> None:
    assert parse_comma_separated("1,None,3", int_or_none) == [1, None, 3]


def test_parse_comma_separated_float_inf() -> None:
    result = parse_comma_separated("inf", float)
    assert result == [float("inf")]


def test_parse_comma_separated_mixed_floats() -> None:
    result = parse_comma_separated("10,inf,0.5", float)
    assert result == [10.0, float("inf"), 0.5]


# ---- is_castable_to_int ----


def test_is_castable_to_int_valid() -> None:
    assert is_castable_to_int("42") is True
    assert is_castable_to_int("-1") is True
    assert is_castable_to_int("0") is True


def test_is_castable_to_int_invalid() -> None:
    assert is_castable_to_int("hello") is False
    assert is_castable_to_int("3.14") is False
    assert is_castable_to_int("") is False


# ---- int_or_none ----


def test_int_or_none_integer() -> None:
    assert int_or_none("5") == 5
    assert int_or_none("-10") == -10


def test_int_or_none_none_literal() -> None:
    assert int_or_none("none") is None
    assert int_or_none("None") is None
    assert int_or_none("NONE") is None


def test_int_or_none_invalid_raises() -> None:
    with pytest.raises(ValueError):
        int_or_none("abc")


# ---- argmedian ----


def test_argmedian_odd_length() -> None:
    assert argmedian(np.array([10, 30, 20])) == 2


def test_argmedian_single_element() -> None:
    assert argmedian(np.array([7])) == 0


def test_argmedian_even_length_picks_nearest() -> None:
    idx = argmedian(np.array([1, 3, 5, 7]))
    assert idx in (1, 2)


# ---- exceeds_deadline ----


def test_exceeds_deadline_none_is_unbounded() -> None:
    assert exceeds_deadline(3600.0, None) is False


def test_exceeds_deadline_false_when_sleep_fits() -> None:
    deadline = time.perf_counter_ns() + int(10 * 1e9)
    assert exceeds_deadline(0.1, deadline) is False


def test_exceeds_deadline_true_when_sleep_overshoots() -> None:
    deadline = time.perf_counter_ns() + int(0.05 * 1e9)
    assert exceeds_deadline(5.0, deadline) is True


def test_exceeds_deadline_zero_seconds_does_not_overshoot() -> None:
    deadline = time.perf_counter_ns() + int(0.001 * 1e9)
    assert exceeds_deadline(0.0, deadline) is False


# ---- wait_for_server_ready ----


def _mock_response(status: int) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value.status = status
    response.__exit__.return_value = False
    return response


def test_wait_for_server_ready_returns_on_200(mocker: MockerFixture) -> None:
    mocker.patch("urllib.request.urlopen", return_value=_mock_response(200))
    mocker.patch("time.monotonic", side_effect=[100.0, 101.5])
    sleep = mocker.patch("time.sleep")

    elapsed = wait_for_server_ready(
        "localhost", 8000, timeout_s=60, backend="modular"
    )

    assert elapsed == pytest.approx(1.5)
    sleep.assert_not_called()


@pytest.mark.parametrize(
    "first_attempt",
    [503, 404, 204, ConnectionRefusedError()],
)
def test_wait_for_server_ready_polls_past_non_ready(
    mocker: MockerFixture,
    first_attempt: int | Exception,
) -> None:
    urlopen = mocker.patch("urllib.request.urlopen")
    if isinstance(first_attempt, Exception):
        urlopen.side_effect = [first_attempt, _mock_response(200)]
    else:
        urlopen.side_effect = [
            _mock_response(first_attempt),
            _mock_response(200),
        ]
    mocker.patch("time.monotonic", side_effect=[100.0, 101.0, 105.0])
    sleep = mocker.patch("time.sleep")

    elapsed = wait_for_server_ready(
        "localhost", 8000, timeout_s=60, backend="modular"
    )

    assert elapsed == pytest.approx(5.0)
    assert urlopen.call_count == 2
    sleep.assert_called_once_with(5.0)


def test_wait_for_server_ready_raises_on_timeout(
    mocker: MockerFixture,
) -> None:
    mocker.patch("urllib.request.urlopen", return_value=_mock_response(503))
    mocker.patch("time.monotonic", side_effect=[100.0, 200.0])
    sleep = mocker.patch("time.sleep")

    with pytest.raises(RuntimeError, match="not ready"):
        wait_for_server_ready("localhost", 8000, timeout_s=5, backend="modular")

    sleep.assert_not_called()


@pytest.mark.parametrize(
    ("first_attempt", "expects_raise"),
    [
        (200, False),
        (503, True),
        (ConnectionRefusedError(), True),
    ],
)
def test_wait_for_server_ready_zero_timeout_tries_once(
    mocker: MockerFixture,
    first_attempt: int | Exception,
    expects_raise: bool,
) -> None:
    urlopen = mocker.patch("urllib.request.urlopen")
    if isinstance(first_attempt, Exception):
        urlopen.side_effect = [first_attempt]
    else:
        urlopen.return_value = _mock_response(first_attempt)
    mocker.patch("time.monotonic", side_effect=[100.0, 100.0])
    sleep = mocker.patch("time.sleep")

    if expects_raise:
        with pytest.raises(RuntimeError):
            wait_for_server_ready(
                "localhost", 8000, timeout_s=0, backend="modular"
            )
    else:
        wait_for_server_ready("localhost", 8000, timeout_s=0, backend="modular")

    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_wait_for_server_ready_aborts_when_liveness_fails(
    mocker: MockerFixture,
) -> None:
    """A dead server aborts immediately instead of polling until timeout."""
    urlopen = mocker.patch("urllib.request.urlopen")
    urlopen.side_effect = ConnectionRefusedError()
    # A long timeout: without the liveness check this would poll for ~hours.
    mocker.patch("time.monotonic", side_effect=[100.0, 100.0])
    sleep = mocker.patch("time.sleep")

    with pytest.raises(RuntimeError, match="exited before"):
        wait_for_server_ready(
            "localhost",
            8000,
            timeout_s=120 * 60,
            backend="modular",
            liveness_check=lambda: False,
        )

    # Aborted on the first failed poll — no sleep, no deadline wait.
    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_wait_for_server_ready_live_liveness_does_not_interfere(
    mocker: MockerFixture,
) -> None:
    """A liveness check that stays True lets normal polling proceed to 200."""
    urlopen = mocker.patch("urllib.request.urlopen")
    urlopen.side_effect = [
        ConnectionRefusedError(),
        _mock_response(200),
    ]
    mocker.patch("time.monotonic", side_effect=[100.0, 101.0, 105.0])
    sleep = mocker.patch("time.sleep")
    liveness = mocker.Mock(return_value=True)

    elapsed = wait_for_server_ready(
        "localhost",
        8000,
        timeout_s=60,
        backend="modular",
        liveness_check=liveness,
    )

    assert elapsed == pytest.approx(5.0)
    assert urlopen.call_count == 2
    # Checked once after the first failed poll; the second poll returned 200.
    liveness.assert_called_once_with()
    sleep.assert_called_once_with(5.0)


def test_wait_for_server_ready_base_url_uses_url_and_bearer_auth(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    urlopen = mocker.patch(
        "urllib.request.urlopen", return_value=_mock_response(200)
    )
    mocker.patch("time.monotonic", side_effect=[100.0, 101.0])

    wait_for_server_ready(
        "localhost",
        8000,
        timeout_s=60,
        backend="modular",
        base_url="https://example.com/",
    )

    request = urlopen.call_args.args[0]
    assert request.full_url == "https://example.com/health"
    assert request.get_header("Authorization") == "Bearer test-key"


def test_wait_for_server_ready_no_base_url_sends_no_auth(
    mocker: MockerFixture,
) -> None:
    urlopen = mocker.patch(
        "urllib.request.urlopen", return_value=_mock_response(200)
    )
    mocker.patch("time.monotonic", side_effect=[100.0, 101.0])

    wait_for_server_ready("localhost", 8000, timeout_s=60, backend="modular")

    request = urlopen.call_args.args[0]
    assert request.full_url == "http://localhost:8000/health"
    assert request.get_header("Authorization") is None
