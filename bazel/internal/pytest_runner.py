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

import argparse
import faulthandler
import importlib
import os
import platform
import shlex
import signal
import sys
import sysconfig
from pathlib import Path

import pytest

# dumps stack traces when bazel kills a hung / slow test
faulthandler.register(signal.SIGTERM)


def __build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-k")
    parser.add_argument("-n")
    return parser


def __exclude_env(key: str) -> bool:
    return key.startswith("BASH_FUNC") or key == "MODULAR_PYTEST_DEBUG"


def __set_torch_memory_limit() -> None:
    if size := os.getenv("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE"):
        if size == "0":
            return

        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError:
            return

        if not torch.cuda.is_available():
            return

        # torch requires the fraction of total memory available on the GPU that
        # torch is allowed to use. Calculate that fraction based on the total
        # memory and the memory request of the specific test. Give the
        # remaining to DeviceContext.
        _, total_memory = torch.cuda.mem_get_info()
        total_requested_fraction = min(float(size) / total_memory, 1.0)
        requested_torch_percent = min(
            1.0,
            max(0.0, float(os.getenv("MODULAR_TORCH_MEMORY_PERCENT", 0.10))),
        )

        torch_fraction = total_requested_fraction * requested_torch_percent
        torch.cuda.set_per_process_memory_fraction(torch_fraction)

        our_total_memory = int(float(size) * (1 - requested_torch_percent))
        os.environ["MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE"] = str(
            our_total_memory
        )


def run():  # noqa: ANN201
    # Allow opt-out of using $TEST_TMPDIR for HF_HOME for performance reasons in
    # special cases
    os.environ["HF_MODULES_CACHE"] = os.environ["TEST_TMPDIR"]
    if os.environ.get("HF_ESCAPES_SANDBOX") != "1":
        os.environ["HF_HOME"] = os.environ["TEST_TMPDIR"]

    # Setup python for nested mojo runs
    os.environ["MOJO_PYTHON"] = sys.executable
    os.environ["MOJO_PYTHON_LIBRARY"] = (
        Path(sys.executable).resolve().parent.parent
        / "lib"
        / sysconfig.get_config_var("INSTSONAME")
    ).as_posix()

    args = list(sys.argv)[1:]
    filter = os.environ.get("TESTBRIDGE_TEST_ONLY")
    if filter:
        args.extend(["-k", filter])

    if os.environ.get("MODULAR_PYTEST_DEBUG"):
        print("\033[31mLLDB SHOULD NOW LAUNCH AUTOMATICALLY\033[0m", flush=True)

        extension = "dylib" if platform.system() == "Darwin" else "so"
        script = "/tmp/lldb.sh"
        with open(script, "w") as f:
            f.write(
                """\
#!/usr/bin/env bash

set -euo pipefail

lldb_bin=bazel-bin/external/+llvm_configure+llvm-project/lldb/lldb
if [[ "${{MODULAR_SYSTEM_LLDB:-}}" == "1" ]]; then
  lldb_bin=lldb
fi

if [[ "${{MODULAR_VSCODE_DEBUG:-}}" == "1" ]]; then
  env {pairs} MODULAR_HOME=$PWD/.derived bazel-bin/KGEN/tools/mojo/mojo debug --vscode -- {args}
elif [[ "${{MODULAR_GDB:-}}" == "1" || "${{MODULAR_ROCGDB:-}}" == "1" ]]; then
  exe=/usr/bin/rocgdb
  if [[ "${{MODULAR_GDB:-}}" == "1" ]]; then
    exe=/usr/bin/gdb
  fi
  env {pairs} \
      "$exe" \
      --eval-command="set cwd {pwd}" \
      --args {args}
elif [[ "${{MODULAR_XCTRACE:-}}" == "1" ]]; then
  env {pairs} \
    /usr/bin/xctrace record --template 'Time Profiler' --launch --target-stdout - \
    -- {args}
elif [[ "${{MODULAR_RR:-}}" == "1" ]]; then
  env {pairs} \
    /usr/bin/rr record \
    -- {args}
else
  env {pairs} \
      "$lldb_bin" \
      --one-line-before-file 'plugin load bazel-bin/Mojo/libMojoLLDB.{extension}' \
      --one-line-before-file 'settings set target.launch-working-dir {pwd}' \
      -- {args}
fi
""".format(
                    extension=extension,
                    pwd=os.getcwd(),
                    pairs=" ".join(
                        f"{k}={shlex.quote(v)}"
                        for k, v in os.environ.items()
                        if not __exclude_env(k)
                    ),
                    args=" ".join([sys.executable] + sys.argv),
                )
            )
        os.chmod(script, 0o777)
        sys.exit(0)

    if "MODULAR_RUNNING_TESTS" not in os.environ:
        raise SystemExit(
            "\033[31merror\033[0m: use 'bazel test' instead of 'bazel run' to run tests"
        )

    __set_torch_memory_limit()
    namespace, unknown_args = __build_parser().parse_known_args(args)
    xml_file = os.environ["XML_OUTPUT_FILE"]
    if Path(xml_file).exists():
        os.chmod(xml_file, mode=0o600)
    pytest_args = [
        f"--junitxml={xml_file}",
        "-o",
        "xfail_strict=true",
        "-o",
        "filterwarnings=error::pytest.PytestUnhandledCoroutineWarning",
        "-o",
        f"junit_suite_name={os.environ['TEST_TARGET']}",
    ]

    if total_shards := os.environ.get("TEST_TOTAL_SHARDS"):
        Path(os.environ["TEST_SHARD_STATUS_FILE"]).touch()
        shard_id = os.environ["TEST_SHARD_INDEX"]
        pytest_args.extend(
            [
                f"--shard-id={shard_id}",
                f"--num-shards={total_shards}",
            ]
        )

    if importlib.util.find_spec("pytest_asyncio"):
        pytest_args.append("--asyncio-mode=auto")

    if namespace.k:
        pytest_args.extend(["-k", namespace.k])
    elif namespace.n:  # Skip xdist when filtering
        pytest_args.extend(["-n", namespace.n])

    exit_code = pytest.main(pytest_args + unknown_args)
    # https://docs.pytest.org/en/6.2.x/usage.html#possible-exit-codes
    if exit_code == 5:  # no tests were collected
        # Print an error:
        if namespace.k:
            print(
                f"\033[31mERROR\033[0m: No tests were run matching the filter: '{namespace.k}'"
            )
        else:
            print("\033[31mERROR\033[0m: No tests were run.")
    return exit_code


if __name__ == "__main__":
    sys.exit(run())
