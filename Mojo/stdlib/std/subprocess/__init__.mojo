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
"""Execute external processes and commands.

The `subprocess` package provides utilities for spawning and interacting with
external processes. It enables running shell commands, capturing their output,
and integrating external tools into Mojo programs. This package handles process
execution, output capture, and resource cleanup automatically.

Use this package when you need to execute shell commands, integrate with
external tools, or automate system tasks from within Mojo code.
"""

from .subprocess import run
