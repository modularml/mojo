# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
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

# This sample prints the current host system information using APIs from the
# sys module.

from runtime.llcl import num_cores
from sys.info import (
    os_is_linux,
    os_is_windows,
    os_is_macos,
    has_sse4,
    has_avx,
    has_avx2,
    has_avx512f,
    has_vnni,
    has_neon,
    is_apple_m1,
    has_intel_amx,
    _current_target,
    _current_cpu,
    _triple_attr,
)


def main():
    var os = ""
    if os_is_linux():
        os = "linux"
    elif os_is_macos():
        os = "macOS"
    else:
        os = "windows"
    let cpu = String(_current_cpu())
    let arch = String(_triple_attr())
    var cpu_features = String("")
    if has_sse4():
        cpu_features += " sse4"
    if has_avx():
        cpu_features += " avx"
    if has_avx2():
        cpu_features += " avx2"
    if has_avx512f():
        cpu_features += " avx512f"
    if has_vnni():
        if has_avx512f():
            cpu_features += " avx512_vnni"
        else:
            cpu_features += " avx_vnni"
    if has_intel_amx():
        cpu_features += " intel_amx"
    if has_neon():
        cpu_features += " neon"
    if is_apple_m1():
        cpu_features += " Apple M1"

    print("System information: ")
    print("    OS          : ", os)
    print("    CPU         : ", cpu)
    print("    Arch        : ", arch)
    print("    Num Cores   : ", num_cores())
    print("    CPU Features:", cpu_features)
