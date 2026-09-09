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

"""Tests for MAX Serve configuration."""

import os

import pytest
from fastapi import FastAPI
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.serve.api_server import fastapi_config
from max.serve.config import Settings


def test_deprecated_dispatcher_config_fails() -> None:
    """Test that using deprecated MAX_SERVE_DISPATCHER_CONFIG fails loudly."""
    # Set the deprecated environment variable
    os.environ["MAX_SERVE_DISPATCHER_CONFIG"] = "tcp://localhost:5555"

    try:
        # Attempting to create Settings should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            Settings()

        # Verify the error message contains helpful information
        error_message = str(exc_info.value)
        assert "MAX_SERVE_DISPATCHER_CONFIG" in error_message
        assert "deprecated" in error_message.lower()
        assert "MAX_SERVE_DI_BIND_ADDRESS" in error_message
        assert "CLIN-608" in error_message
    finally:
        # Clean up the environment variable
        del os.environ["MAX_SERVE_DISPATCHER_CONFIG"]


def test_di_bind_address_works_without_deprecated_var() -> None:
    """Test that MAX_SERVE_DI_BIND_ADDRESS works correctly when deprecated var is not set."""
    # Ensure the deprecated variable is not set
    if "MAX_SERVE_DISPATCHER_CONFIG" in os.environ:
        del os.environ["MAX_SERVE_DISPATCHER_CONFIG"]

    # Test with default value
    settings = Settings()
    assert settings.di_bind_address == "tcp://127.0.0.1:5555"

    # Test with custom value via environment variable
    os.environ["MAX_SERVE_DI_BIND_ADDRESS"] = "tcp://0.0.0.0:6666"
    try:
        settings = Settings()
        assert settings.di_bind_address == "tcp://0.0.0.0:6666"
    finally:
        del os.environ["MAX_SERVE_DI_BIND_ADDRESS"]


def test_deprecated_dispatcher_config_fails_even_with_valid_di_bind_address() -> (
    None
):
    """Test that deprecated variable causes failure even if new variable is also set."""
    os.environ["MAX_SERVE_DISPATCHER_CONFIG"] = "tcp://old:5555"
    os.environ["MAX_SERVE_DI_BIND_ADDRESS"] = "tcp://new:6666"

    try:
        with pytest.raises(ValueError) as exc_info:
            Settings()

        error_message = str(exc_info.value)
        assert "MAX_SERVE_DISPATCHER_CONFIG" in error_message
        assert "deprecated" in error_message.lower()
    finally:
        if "MAX_SERVE_DISPATCHER_CONFIG" in os.environ:
            del os.environ["MAX_SERVE_DISPATCHER_CONFIG"]
        if "MAX_SERVE_DI_BIND_ADDRESS" in os.environ:
            del os.environ["MAX_SERVE_DI_BIND_ADDRESS"]


def test_eplb_profile_default_is_false() -> None:
    """EP profiling must default to off — opt-in only."""
    if "MAX_SERVE_EPLB_PROFILE" in os.environ:
        del os.environ["MAX_SERVE_EPLB_PROFILE"]

    settings = Settings()
    assert settings.eplb_profile is False


def test_eplb_profile_can_be_enabled_via_env() -> None:
    """MAX_SERVE_EPLB_PROFILE=1 enables profiling."""
    os.environ["MAX_SERVE_EPLB_PROFILE"] = "1"
    try:
        settings = Settings()
        assert settings.eplb_profile is True
    finally:
        del os.environ["MAX_SERVE_EPLB_PROFILE"]


def test_eplb_profile_can_be_set_by_field_name() -> None:
    """Direct init by field name (eplb_profile=True) works."""
    if "MAX_SERVE_EPLB_PROFILE" in os.environ:
        del os.environ["MAX_SERVE_EPLB_PROFILE"]

    settings = Settings(eplb_profile=True)
    assert settings.eplb_profile is True


def test_pipeline_runtime_config_eplb_profile_mirrors_env() -> None:
    """PipelineRuntimeConfig.eplb_profile reads MAX_SERVE_EPLB_PROFILE
    so pipeline code has the same view as Settings."""

    if "MAX_SERVE_EPLB_PROFILE" in os.environ:
        del os.environ["MAX_SERVE_EPLB_PROFILE"]
    assert PipelineRuntimeConfig().eplb_profile is False

    os.environ["MAX_SERVE_EPLB_PROFILE"] = "1"
    try:
        assert PipelineRuntimeConfig().eplb_profile is True
    finally:
        del os.environ["MAX_SERVE_EPLB_PROFILE"]


def test_graceful_shutdown_timeout_default() -> None:
    from max.serve.config import Settings

    settings = Settings()
    assert settings.graceful_shutdown_timeout_s == 5


def test_graceful_shutdown_timeout_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from max.serve.config import Settings

    monkeypatch.setenv("MAX_SERVE_GRACEFUL_SHUTDOWN_TIMEOUT_S", "120")
    settings = Settings()
    assert settings.graceful_shutdown_timeout_s == 120


def test_fastapi_config_wires_graceful_shutdown_timeout() -> None:
    """The setting must reach uvicorn's timeout_graceful_shutdown.

    This is the load-bearing property of the change: how long uvicorn waits for
    in-flight requests on SIGTERM is governed by graceful_shutdown_timeout_s. The
    graceful shutdown itself is uvicorn's behavior; we only verify the wiring,
    with no model worker or server start required.
    """
    config = fastapi_config(FastAPI(), Settings(graceful_shutdown_timeout_s=42))
    assert config.timeout_graceful_shutdown == 42


def test_http_keepalive_timeout_outlives_pooling_clients() -> None:
    """The default must sit above a pooling client's idle-connection timeout.

    Go's ``http.Transport`` idles pooled connections out at 90s by default. If
    the server closes first, a client reusing a pooled connection races that
    close and sees a TCP reset instead of a response -- and cannot retry it,
    because a proxied POST body is not replayable.
    """
    assert Settings().http_keepalive_timeout_s > 90


def test_http_keepalive_timeout_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_SERVE_HTTP_KEEPALIVE_TIMEOUT_S", "300")
    assert Settings().http_keepalive_timeout_s == 300


def test_fastapi_config_wires_http_keepalive_timeout() -> None:
    config = fastapi_config(FastAPI(), Settings(http_keepalive_timeout_s=137))
    assert config.timeout_keep_alive == 137
