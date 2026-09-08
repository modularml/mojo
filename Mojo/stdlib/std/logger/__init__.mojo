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
"""Logging with configurable severity levels.

The `logger` package provides a flexible logging system with multiple severity
levels for debugging and monitoring applications. It supports configurable log
levels (`TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`), colored output,
customizable formatting, and environment-based configuration. The logging level
can be set via the LOGGING_LEVEL environment variable to control message
verbosity.

Use this package for debugging, monitoring application behavior, error
reporting, or adding instrumentation to track program execution. Configure the
logging level in development for detailed output, then reduce verbosity in
production for performance.
"""

from .logger import Level, Logger
