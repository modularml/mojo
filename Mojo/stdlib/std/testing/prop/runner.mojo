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
from ._shrinking._shrinker import Shrinker, ShrinkResult


struct PropTestConfig(Copyable):
    """A configuration for a property test."""

    var runs: Int
    """The number of successful test runs to achieve before stopping the test."""

    var seed: Int
    """The seed for the random number generator."""

    var shrink: Bool
    """Run test case reduction on failed test."""

    var max_shrink_steps: Int
    """Max number of successful shrink steps to run in test case reduction."""

    def __init__(
        out self,
        *,
        runs: Int = 100,
        seed: Optional[Int] = None,
        shrink: Bool = True,
        max_shrink_steps: Int = 1000,
    ):
        """Construct a new property test configuration.

        Args:
            runs: The number of successful test runs to achieve before stopping the test.
            seed: The seed for the random number generator.
            shrink: Whether to run test case reduction on failed tests.
            max_shrink_steps: The maximum number of successful shrink steps to run.
        """
        self.runs = runs
        self.seed = seed.or_else(Int(perf_counter_ns()))
        self.shrink = shrink
        self.max_shrink_steps = max_shrink_steps


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
                f(value.copy())
            except e:
                if (
                    len(rng.history) == 0
                    or not self._config.shrink
                    or self._config.max_shrink_steps == 0
                ):
                    # If the history is empty, we have no information to shrink,
                    # so we just raise the error.
                    raise _PropTestError(
                        shrink_result=ShrinkResult(value^, 0),
                        runs=i,
                        seed=self._config.seed,
                        error=e^,
                    )

                var shrinker = Shrinker[type_of(strategy), f](
                    strategy^,
                    rng.history.copy(),
                    e.copy(),
                    self._config.max_shrink_steps,
                )
                var shrink_result = shrinker.shrink()
                raise _PropTestError(
                    shrink_result=shrink_result^,
                    original_value=value^,
                    runs=i,
                    seed=self._config.seed,
                    error=e^,
                )


struct _PropTestError[T: Copyable & Deinitable & Writable, //](
    Copyable, Writable
):
    var runs: Int
    var seed: Int
    var error: Error
    var shrink_result: ShrinkResult[Self.T]
    var original_value: Optional[Self.T]

    def __init__(
        out self,
        *,
        var shrink_result: ShrinkResult[Self.T],
        runs: Int,
        seed: Int,
        var error: Error = {},
        var original_value: Optional[Self.T] = None,
    ):
        self.runs = runs
        self.seed = seed
        self.error = error^
        self.shrink_result = shrink_result^
        self.original_value = original_value^

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            t"PropTest failed after {self.runs} runs (seed: {self.seed})\n"
        )
        writer.write(t"Falsifying input: {self.shrink_result.value}\n")
        if self.original_value:
            ref value = self.original_value.value()
            writer.write(
                t"(shrunk from {value} in {self.shrink_result.runs} shrink"
                t" steps)\n"
            )
        writer.write(t"Error: {self.error}")
