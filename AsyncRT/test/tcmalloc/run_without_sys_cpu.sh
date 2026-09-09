#!/bin/bash
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
#
# A compiled Mojo program must start even when /sys/devices/system/cpu is
# hidden, as may be the case in a restricted sandbox. This script masks it via a
# throwaway user+mount namespace.
set -uo pipefail

# Bazel passes the Mojo binary as $(rootpath ...); resolve it to an absolute path.
bin="${1:?usage: run_without_sys_cpu.sh <binary>}"
bin="$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")"

# Masking /sys needs unprivileged user namespaces. Fail loudly if they are unavailable.
unshare -rm true 2>/dev/null || { echo "FAIL: user namespaces unavailable"; exit 1; }

unshare -rm sh -c "mount -t tmpfs none /sys/devices/system/cpu && exec '$bin'"
rc=$?
[ "$rc" -eq 0 ] || { echo "FAIL: runtime aborted (exit $rc) under masked /sys"; exit 1; }
echo "PASS: ran under masked /sys"
