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
"""Unit tests for the unified DebugConfig system.

These tests validate the DebugConfig class exposed on InferenceSession.debug
and Graph.debug. They cover:
- Boolean and value property defaults and setters
- Meta 'sensible' debug mode
- Individual overrides after meta mode
- MODULAR_DEBUG environment variable parsing
- Class-level access (without an instance)
"""

import os
import subprocess
import sys
import textwrap
from collections.abc import Generator
from pathlib import Path

import pytest
from max.engine import InferenceSession, PrintStyle
from max.graph import Graph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_debug_config() -> Generator[None]:
    """Reset the DebugConfig singleton after each test to avoid cross-test pollution."""
    yield
    InferenceSession.debug.reset()


# ---------------------------------------------------------------------------
# Boolean property defaults and setters
# ---------------------------------------------------------------------------


class TestBooleanPropertyDefaults:
    """Each boolean debug property should default to False."""

    def test_nan_check_default(self) -> None:
        assert InferenceSession.debug.nan_check is False

    def test_uninitialized_read_check_default(self) -> None:
        assert InferenceSession.debug.uninitialized_read_check is False

    def test_device_sync_mode_default(self) -> None:
        assert InferenceSession.debug.device_sync_mode is False

    def test_stack_trace_on_error_default(self) -> None:
        assert InferenceSession.debug.stack_trace_on_error is False

    def test_stack_trace_on_crash_default(self) -> None:
        assert InferenceSession.debug.stack_trace_on_crash is False


class TestBooleanPropertySetters:
    """Boolean properties should be settable to True and back to False."""

    def test_nan_check_enable(self) -> None:
        InferenceSession.debug.nan_check = True
        assert InferenceSession.debug.nan_check is True

    def test_nan_check_disable(self) -> None:
        InferenceSession.debug.nan_check = True
        InferenceSession.debug.nan_check = False
        assert InferenceSession.debug.nan_check is False

    def test_uninitialized_read_check_enable(self) -> None:
        InferenceSession.debug.uninitialized_read_check = True
        assert InferenceSession.debug.uninitialized_read_check is True

    def test_device_sync_mode_enable(self) -> None:
        InferenceSession.debug.device_sync_mode = True
        assert InferenceSession.debug.device_sync_mode is True

    def test_stack_trace_on_error_enable(self) -> None:
        InferenceSession.debug.stack_trace_on_error = True
        assert InferenceSession.debug.stack_trace_on_error is True

    def test_stack_trace_on_crash_enable(self) -> None:
        InferenceSession.debug.stack_trace_on_crash = True
        assert InferenceSession.debug.stack_trace_on_crash is True


# ---------------------------------------------------------------------------
# Value property defaults and setters
# ---------------------------------------------------------------------------


class TestValuePropertyDefaults:
    """Value properties should have sensible defaults."""

    def test_op_log_level_default(self) -> None:
        assert InferenceSession.debug.op_log_level in ("", "notset")

    def test_assert_level_default(self) -> None:
        assert InferenceSession.debug.assert_level in ("", "none")

    def test_print_style_default(self) -> None:
        # Default ("") maps to PrintStyle.NONE ("disables debug output").
        assert InferenceSession.debug.print_style == PrintStyle.NONE

    def test_ir_output_dir_default(self) -> None:
        assert InferenceSession.debug.ir_output_dir == ""


class TestValuePropertySetters:
    """Value properties should accept and return the assigned value."""

    def test_op_log_level_set(self) -> None:
        InferenceSession.debug.op_log_level = "DEBUG"
        assert InferenceSession.debug.op_log_level == "DEBUG"

    def test_assert_level_set(self) -> None:
        InferenceSession.debug.assert_level = "all"
        assert InferenceSession.debug.assert_level == "all"

    def test_print_style_set(self) -> None:
        InferenceSession.debug.print_style = PrintStyle.COMPACT
        assert InferenceSession.debug.print_style == PrintStyle.COMPACT

    def test_ir_output_dir_set(self, tmp_path: Path) -> None:
        dir_path = str(tmp_path)
        InferenceSession.debug.ir_output_dir = dir_path
        assert InferenceSession.debug.ir_output_dir == dir_path


def _run_with_env(
    env_overrides: dict[str, str], check_script: str
) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a subprocess with extra env vars set."""
    env = os.environ.copy()
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(check_script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# Source tracebacks (lives on Graph.debug)
# ---------------------------------------------------------------------------


class TestSourceTracebacks:
    """Source tracebacks are configured via Graph.debug, not InferenceSession.debug."""

    def test_source_tracebacks_default(self) -> None:
        assert Graph.debug.source_tracebacks is False

    def test_source_tracebacks_enable(self) -> None:
        Graph.debug.source_tracebacks = True
        assert Graph.debug.source_tracebacks is True

    def test_source_tracebacks_disable(self) -> None:
        Graph.debug.source_tracebacks = True
        Graph.debug.source_tracebacks = False
        assert Graph.debug.source_tracebacks is False


# ---------------------------------------------------------------------------
# Meta 'sensible' debug mode
# ---------------------------------------------------------------------------


class TestMetaDebugMode:
    """Setting session.debug.sensible_mode = True should enable a sensible set of debug options."""

    def test_mode_enables_nan_check(self) -> None:
        InferenceSession.debug.sensible_mode = True
        assert InferenceSession.debug.nan_check is True

    def test_mode_does_not_enable_uninitialized_read_check(self) -> None:
        InferenceSession.debug.sensible_mode = True
        assert InferenceSession.debug.uninitialized_read_check is False

    def test_mode_enables_assert_level_all(self) -> None:
        InferenceSession.debug.sensible_mode = True
        assert InferenceSession.debug.assert_level == "all"

    def test_mode_enables_device_sync_mode(self) -> None:
        InferenceSession.debug.sensible_mode = True
        assert InferenceSession.debug.device_sync_mode is True

    def test_mode_enables_stack_trace_on_error(self) -> None:
        InferenceSession.debug.sensible_mode = True
        assert InferenceSession.debug.stack_trace_on_error is True

    def test_mode_enables_stack_trace_on_crash(self) -> None:
        InferenceSession.debug.sensible_mode = True
        assert InferenceSession.debug.stack_trace_on_crash is True

    def test_mode_enables_source_tracebacks(self) -> None:
        InferenceSession.debug.sensible_mode = True
        assert Graph.debug.source_tracebacks is True


# ---------------------------------------------------------------------------
# Individual overrides beat meta mode
# ---------------------------------------------------------------------------


class TestOverridesAfterMetaMode:
    """Setting an individual property after mode=True should override the meta value."""

    def test_override_nan_check_false(self) -> None:
        InferenceSession.debug.sensible_mode = True
        InferenceSession.debug.nan_check = False
        assert InferenceSession.debug.nan_check is False

    def test_override_assert_level(self) -> None:
        InferenceSession.debug.sensible_mode = True
        InferenceSession.debug.assert_level = "safe"
        assert InferenceSession.debug.assert_level == "safe"

    def test_override_device_sync_mode_false(self) -> None:
        InferenceSession.debug.sensible_mode = True
        InferenceSession.debug.device_sync_mode = False
        assert InferenceSession.debug.device_sync_mode is False

    def test_override_stack_trace_on_error_false(self) -> None:
        InferenceSession.debug.sensible_mode = True
        InferenceSession.debug.stack_trace_on_error = False
        assert InferenceSession.debug.stack_trace_on_error is False

    def test_override_source_tracebacks_false(self) -> None:
        InferenceSession.debug.sensible_mode = True
        Graph.debug.source_tracebacks = False
        assert Graph.debug.source_tracebacks is False


# ---------------------------------------------------------------------------
# MODULAR_DEBUG env var parsing (subprocess isolation)
# ---------------------------------------------------------------------------


def _run_debug_env_check(
    env_value: str, check_script: str
) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a subprocess with MODULAR_DEBUG set."""
    return _run_with_env({"MODULAR_DEBUG": env_value}, check_script)


class TestModularDebugEnvVar:
    """MODULAR_DEBUG env var should configure debug options at startup."""

    def test_nan_check_flag(self) -> None:
        result = _run_debug_env_check(
            "nan-check",
            """\
            from max.engine import InferenceSession
            assert InferenceSession.debug.nan_check is True, (
                f"Expected nan_check=True, got {InferenceSession.debug.nan_check}"
            )
            print("PASS")
            """,
        )
        assert result.returncode == 0, (
            f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PASS" in result.stdout

    def test_device_sync_mode_flag(self) -> None:
        result = _run_debug_env_check(
            "device-sync-mode",
            """\
            from max.engine import InferenceSession
            assert InferenceSession.debug.device_sync_mode is True, (
                f"Expected device_sync_mode=True, got {InferenceSession.debug.device_sync_mode}"
            )
            print("PASS")
            """,
        )
        assert result.returncode == 0, (
            f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PASS" in result.stdout

    def test_multiple_flags_comma_separated(self) -> None:
        result = _run_debug_env_check(
            "nan-check,device-sync-mode",
            """\
            from max.engine import InferenceSession
            assert InferenceSession.debug.nan_check is True, (
                f"Expected nan_check=True, got {InferenceSession.debug.nan_check}"
            )
            assert InferenceSession.debug.device_sync_mode is True, (
                f"Expected device_sync_mode=True, got {InferenceSession.debug.device_sync_mode}"
            )
            print("PASS")
            """,
        )
        assert result.returncode == 0, (
            f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PASS" in result.stdout

    def test_assert_level_key_value(self) -> None:
        result = _run_debug_env_check(
            "assert-level=all",
            """\
            from max.engine import InferenceSession
            assert InferenceSession.debug.assert_level == "all", (
                f"Expected assert_level='all', got {InferenceSession.debug.assert_level!r}"
            )
            print("PASS")
            """,
        )
        assert result.returncode == 0, (
            f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PASS" in result.stdout

    def test_sensible_meta_flag(self) -> None:
        result = _run_debug_env_check(
            "sensible",
            """\
            from max.engine import InferenceSession
            from max.graph import Graph
            assert InferenceSession.debug.nan_check is True, "nan_check"
            assert InferenceSession.debug.uninitialized_read_check is False, "uninitialized_read_check should NOT be in sensible set"
            assert InferenceSession.debug.device_sync_mode is True, "device_sync_mode"
            assert InferenceSession.debug.stack_trace_on_error is True, "stack_trace_on_error"
            assert InferenceSession.debug.stack_trace_on_crash is True, "stack_trace_on_crash"
            assert InferenceSession.debug.assert_level == "all", "assert_level"
            assert Graph.debug.source_tracebacks is True, "source_tracebacks"
            print("PASS")
            """,
        )
        assert result.returncode == 0, (
            f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PASS" in result.stdout

    def test_unknown_bare_token_warns(self) -> None:
        result = _run_debug_env_check(
            "not-a-real-flag",
            """\
            from max.engine import InferenceSession
            # Touching .debug forces DebugConfig construction and env parsing.
            _ = InferenceSession.debug
            print("PASS")
            """,
        )
        assert result.returncode == 0, (
            f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PASS" in result.stdout
        assert "MODULAR_DEBUG" in result.stderr
        assert "not-a-real-flag" in result.stderr

    def test_unknown_key_value_warns(self) -> None:
        result = _run_debug_env_check(
            "made-up-key=value",
            """\
            from max.engine import InferenceSession
            _ = InferenceSession.debug
            print("PASS")
            """,
        )
        assert result.returncode == 0, (
            f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PASS" in result.stdout
        assert "MODULAR_DEBUG" in result.stderr
        assert "made-up-key" in result.stderr

    def test_known_flags_do_not_warn(self) -> None:
        result = _run_debug_env_check(
            "nan-check,assert-level=all",
            """\
            from max.engine import InferenceSession
            _ = InferenceSession.debug
            print("PASS")
            """,
        )
        assert result.returncode == 0, (
            f"Script failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PASS" in result.stdout
        assert "MODULAR_DEBUG" not in result.stderr


# ---------------------------------------------------------------------------
# Class-level access (no instance required)
# ---------------------------------------------------------------------------


class TestClassLevelAccess:
    """DebugConfig should be accessible as a class attribute, not just on instances."""

    def test_session_debug_accessible_without_instance(self) -> None:
        # Should not raise AttributeError
        debug = InferenceSession.debug
        assert debug is not None

    def test_graph_debug_accessible_without_instance(self) -> None:
        debug = Graph.debug
        assert debug is not None

    def test_instance_and_class_share_state(self) -> None:
        session = InferenceSession()
        InferenceSession.debug.nan_check = True
        assert session.debug.nan_check is True

    def test_class_level_set_reflects_on_instance(self) -> None:
        session = InferenceSession()
        InferenceSession.debug.device_sync_mode = True
        assert session.debug.device_sync_mode is True
