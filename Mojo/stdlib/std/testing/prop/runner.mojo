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
"""Implements the property test runner and configuration."""

from .random import Rng
from .strategy import Strategy
from std.time import perf_counter_ns


struct PropTestConfig(Copyable):
    """A configuration for a property test."""

    var runs: Int
    """The number of successful test runs to achieve before stopping the test."""

    var seed: Int
    """The seed for the random number generator."""

    def __init__(out self, *, runs: Int = 100, seed: Optional[Int] = None):
        """Construct a new property test configuration.

        Args:
            runs: The number of successful test runs to achieve before stopping the test.
            seed: The seed for the random number generator.
        """
        self.runs = runs
        self.seed = seed.or_else(Int(perf_counter_ns()))


struct PropTest(Movable):
    """A property test runner."""

    var _config: PropTestConfig

    def __init__(out self):
        """Construct a new property test runner with the default configuration.

        Returns:
            A new property test runner with the default configuration.
        """
        self = Self(config=PropTestConfig())

    def __init__(out self, *, var config: PropTestConfig):
        """Construct a new property test runner.

        Args:
            config: The configuration for the property test.
        """
        self._config = config^

    # TODO(MOCO 4614) - change this to be a closure argument rather than thin
    def test[
        StrategyType: Strategy,
        //,
        f: def(var StrategyType.Value) thin raises,
    ](self, var strategy: StrategyType) raises:
        """Run a property test with the given strategy.

        Parameters:
            StrategyType: The strategy type to use for the property test.
            f: The function to test.

        Args:
            strategy: The strategy value to use for the property test.

        Raises:
            An error if the property test fails.
        """
        var rng = Rng(seed=self._config.seed)
        for i in range(self._config.runs):
            var value = strategy.value(rng)
            try:
                f(value^)
            except e:
                raise Error(
                    _PropTestError(runs=i, seed=self._config.seed, error=e^)
                )


struct _PropTestError(Copyable, Writable):
    var runs: Int
    var seed: Int
    var error: Error

    def __init__(out self, *, runs: Int, seed: Int, var error: Error = {}):
        self.runs = runs
        self.seed = seed
        self.error = error^

    def write_to(self, mut writer: Some[Writer]):
        writer.write("PropTestError: ", "\n")
        writer.write("\tRuns: ", self.runs, "\n")
        writer.write("\tSeed: ", self.seed, "\n")
        writer.write("\tError: ", self.error)
