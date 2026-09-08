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

# Test that all debug level options are accepted by the compiler.
# This test only validates that the options are parsed correctly.
# See mojo-tool/build/mojo_build_debug_level.mojo for tests that
# verify debug info is actually generated/omitted.
# RUN: %mojo --debug-level=none %s
# RUN: %mojo -g0 %s
# RUN: %mojo --debug-level=line-tables %s
# RUN: %mojo -g1 %s
# RUN: %mojo --debug-level=full %s
# RUN: %mojo -g %s
# RUN: %mojo -g2 %s


# CHECK: debug test ok
def main() -> None:
    print("debug test ok")
