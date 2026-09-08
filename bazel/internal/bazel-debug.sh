#!/usr/bin/env bash
##===----------------------------------------------------------------------===##
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
##===----------------------------------------------------------------------===##

set -euo pipefail

# Test cases:
#
# - cc_binary: //Mojo/tools/mojo -- -h
# - cc_test: //Mojo/unittests
# - cc_test's underlying cc_binary: //Mojo/unittests:unittests.debug
# - py_binary: //max/python/max/_entrypoints:pipelines -- generate --model modularai/Llama-3.1-8B-Instruct-GGUF --prompt "I believe the meaning of life is" --max-new-tokens 8 --max-batch-size 4 --quantization-encoding float32
# - py_test: //Support/test/configuration:env_test
# - mojo_binary: Kernels/test/gpu-query
# - mojo_test: Mojo/stdlib/test:builtin/test_math.mojo.test
# - arg parsing: --config=debug-bazel Support/python:unittests --config=debug-bazel -- other args

target=""
default_config="lldb"
before_target=()
for arg in "$@"
do
  if [[ "$arg" != -* ]]; then
    target="$arg"
    shift
    break
  elif [[ "$arg" = --vscode ]]; then
    export MODULAR_VSCODE_DEBUG="1"
    shift
  elif [[ "$arg" = --gdb ]]; then
    export MODULAR_GDB="1"
    shift
  elif [[ "$arg" = --rocgdb ]]; then
    export MODULAR_ROCGDB="1"
    shift
  elif [[ "$arg" = --system-lldb ]]; then
    export MODULAR_SYSTEM_LLDB="1"
    shift
  elif [[ "$arg" = --xctrace ]]; then
    export MODULAR_XCTRACE="1"
    shift
  elif [[ "$arg" = --rr ]]; then
    export MODULAR_RR="1"
    shift
  else
    before_target+=("$arg")
    shift
  fi
done

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
repo_root="$(git -C "$script_root" rev-parse --show-toplevel)"
wrapper="$repo_root/bazelw"

export MODULAR_LLDB_PWD="$PWD"
current_repo_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ "$repo_root" != "$current_repo_root" ]]; then
  # Only cd if required so relative bazel targets work
  cd "$repo_root"
fi

# py_binary rule //path/to:target
output=$("$wrapper" query "some(labels(actual, $target) union set($target))" --output=label_kind)
kind="${output%% *}"
# Resolve target shorthand and aliases through single query
target=${output##* }

subcommand="run"
run_debug_script=false
if [[ "$kind" == py_binary ]]; then
  # py_binary has an underlying py_repl that we can use for debugging
  target="$target.debug"
elif [[ "$kind" == py_test ]]; then
  # py_test has a custom codepath to dump a debugging script, which should be run after
  run_debug_script=true
  subcommand="test"
  default_config="debug-pytest"
  before_target+=("//bazel:lldb_wrapper")
elif [[ "$kind" == cc_test ]]; then
  # cc_test has an underlying binary that is debuggable with bazel run
  target="$target.debug"
elif [[ "$kind" == mojo_test ]]; then
  # mojo_test has an underlying mojo_binary that is debuggable with bazel run
  target="$target.debug"
elif [[ "$kind" == modular_genrule ]]; then
  echo "error: modular_genrule not currently supported for debugging" >&2
  exit 1
fi

before_target+=("--config=$default_config")
"$wrapper" "$subcommand" "${before_target[@]}" "$target" "$@"
if [[ "$run_debug_script" == "true" ]]; then
  exec /tmp/lldb.sh
fi
