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

from std.reflection import SourceLocation
from std.logger import Level, Logger
from std.testing import TestSuite


# CHECK-LABEL: Test logging at trace level
def test_log_trace() raises:
    print("=== Test logging at trace level")
    var log = Logger[Level.TRACE]()

    # CHECK: {{.*}}TRACE{{.*}}::: hello
    log.trace("hello")

    var log2 = Logger[Level.DEBUG]()
    # CHECK-NOT: {{.*}}TRACE{{.*}}::: hello
    log2.trace("hello")


# CHECK-LABEL: Test logging at info level
def test_log_info() raises:
    print("=== Test logging at info level")
    var log = Logger[Level.INFO]()

    # CHECK-NOT: {{.*}}DEBUG{{.*}}::: hello world
    log.debug("hello", "world")

    # CHECK: {{.*}}INFO{{.*}}::: hello
    log.info("hello")


# CHECK-LABEL: Test no logging by default
def test_log_noset() raises:
    print("=== Test no logging by default")
    var log = Logger()

    # CHECK-NOT: {{.*}}DEBUG{{.*}}::: hello world
    log.debug("hello", "world")

    # CHECK-NOT: {{.*}}INFO{{.*}}::: hello
    log.info("hello")


# CHECK-LABEL: Test logging with prefix
def test_log_with_prefix() raises:
    print("=== Test logging with prefix")

    var log = Logger[Level.TRACE](prefix="[XYZ] ")

    # CHECK: [XYZ]
    # CHECK: hello
    log.trace("hello")


# CHECK-LABEL: Test logging with location
def test_log_with_location() raises:
    print("=== Test logging with location")

    comptime log = Logger[Level.TRACE](prefix="", source_location=True)

    # CHECK: test_logger.mojo:74:14] hello
    log.trace("hello")


# CHECK-LABEL: Test logging with custom location
def test_log_with_custom_location() raises:
    print("=== Test logging with custom location")

    comptime log = Logger[Level.TRACE](prefix="", source_location=True)

    # CHECK: somefile.mojo:42:999] hello
    log.trace("hello", location=SourceLocation(42, 999, "somefile.mojo"))


# CHECK-LABEL: Test logging with sep/end
def test_log_with_sep_end() raises:
    print("=== Test logging with sep/end")

    var log = Logger[Level.TRACE]()

    # CHECK: hello mojo world!!!
    log.trace("hello", "world", sep=" mojo ", end="!!!\n")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
