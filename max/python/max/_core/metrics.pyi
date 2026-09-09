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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

"""In-process metrics collection for the Modular runtime."""

class HistogramSnapshot:
    """
    Statistics accumulated by a Histogram instrument between collect() calls.

    All fields reflect data points recorded since the previous collect().
    """

    @property
    def count(self) -> int:
        """Number of data points recorded."""

    @property
    def sum(self) -> float:
        """Sum of all recorded values."""

    @property
    def min(self) -> float:
        """Minimum recorded value."""

    @property
    def max(self) -> float:
        """Maximum recorded value."""

class CounterSample:
    """
    A single counter reading from a collect() snapshot.

    Counter values are cumulative and never reset.
    """

    @property
    def name(self) -> str:
        """Instrument name."""

    @property
    def value(self) -> int:
        """Cumulative count."""

class GaugeSample:
    """A single gauge value from an instantaneous read."""

    @property
    def name(self) -> str:
        """Instrument name."""

    @property
    def value(self) -> int:
        """Current gauge value."""

class HistogramSample:
    """
    A single histogram reading from a collect() snapshot.

    The underlying accumulator is reset on each collect() call, so ``ss``
    reflects only the interval since the previous collect().
    """

    @property
    def name(self) -> str:
        """Instrument name."""

    @property
    def ss(self) -> HistogramSnapshot:
        """Statistics for this collection interval."""

class CollatedMetrics:
    """Snapshot of all registered instruments, returned by collect()."""

    @property
    def counters(self) -> list[CounterSample]:
        """Counter readings, one entry per registered counter."""

    @property
    def gauges(self) -> list[GaugeSample]:
        """Gauge readings, one entry per registered gauge."""

    @property
    def histograms(self) -> list[HistogramSample]:
        """Histogram readings, one entry per registered histogram."""

def collect() -> CollatedMetrics:
    """
    Snapshot all registered instruments.

    Returns a :class:`CollatedMetrics` containing the current reading of
    every counter and gauge, and a reset snapshot of every histogram.
    Histogram accumulators are cleared on each call; counter and gauge
    values are cumulative.
    """
