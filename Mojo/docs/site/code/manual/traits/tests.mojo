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
# tests.mojo
# Tests for traits.mdx code examples.
# Skip: initial fetch_reading-only DeflectionSensing skeleton
#        (superseded by the fuller trait below), the "Provided
#        methods"/"Required compile-time values" Pausable bullet
#        excerpts (combined into one Pausable trait below, exercised
#        by Timer), the Loggable "Required methods" bullet (identical
#        signature to the trait declared below, no separate
#        declaration needed), the "Shared compile-time constants"
#        DeflectionSensing bullet (duplicate of the constant already
#        on the main trait), Container associated-type snippet (no
#        conforming type shown), Gateway's conflicting-default example
#        exactly as written (`struct Gateway(PowerCycle, Rebootable):
#        pass` doesn't compile by design - the runnable test instead
#        gives Gateway its own restart() to demonstrate the resolution
#        the prose describes).
from std.testing import assert_equal


# --- DeflectionSensing trait (canonical, full version) ---


trait DeflectionSensing:
    def fetch_reading(self) -> Float64:
        ...

    comptime absolute_tolerance: Float64 = 0.05  # This is made up for this example

    def within_tolerance(self) -> Bool:
        return abs(self.fetch_reading()) <= Self.absolute_tolerance


# --- Refining traits: CalibratableDeflectionSensing / EddyCurrentSensor ---
# Given a concrete `reading` field and real method bodies, since the doc
# only shows placeholder comment bodies ("# its implementation").


trait CalibratableDeflectionSensing(DeflectionSensing):
    def calibrate(mut self):
        ...


struct EddyCurrentSensor(CalibratableDeflectionSensing):
    var reading: Float64

    def __init__(out self):
        self.reading = 12.3

    def fetch_reading(self) -> Float64:
        return self.reading

    def calibrate(mut self):
        self.reading = 0.0


def test_eddy_current_sensor() raises:
    var sensor = EddyCurrentSensor()
    assert_equal(sensor.fetch_reading(), 12.3)
    # Inherits within_tolerance() default from DeflectionSensing.
    assert_equal(sensor.within_tolerance(), False)
    sensor.calibrate()
    assert_equal(sensor.fetch_reading(), 0.0)
    assert_equal(sensor.within_tolerance(), True)


# --- Conforming to a trait: CapacitiveSensor ---


@fieldwise_init
struct CapacitiveSensor(Copyable, DeflectionSensing):
    def fetch_reading(self) -> Float64:
        # Not a very good sensor, but a simple example.
        return Float64(21.5)


def test_capacitive_sensor() raises:
    var sensor = CapacitiveSensor()
    assert_equal(sensor.fetch_reading(), 21.5)
    assert_equal(sensor.within_tolerance(), False)


# --- Required comptime members: Pausable / Timer ---
# Combines the "provided method" and "required compile-time value"
# bullet excerpts into the one trait Timer actually exercises.


trait Pausable:
    def pause(self):
        pass

    comptime max_pause_seconds: Float64


@fieldwise_init
struct Timer(Copyable, Pausable):
    comptime max_pause_seconds: Float64 = 30.0

    def pause(self):
        # print("Paused")
        pass


def test_timer_pausable() raises:
    var timer = Timer()
    timer.pause()
    assert_equal(Timer.max_pause_seconds, 30.0)


# --- Parameterizing functions with traits: averaged_poll ---


def averaged_poll[
    SensorType: DeflectionSensing, //  # infer-only
](sensor: SensorType, samples: Int) -> Float64:
    if samples <= 0:
        return 0.0
    var total: Float64 = 0.0
    for _ in range(samples):
        total += sensor.fetch_reading()
    return total / Float64(samples)


def test_averaged_poll() raises:
    var sensor = CapacitiveSensor()
    assert_equal(averaged_poll(sensor, 10), 21.5)


# --- Some[] shorthand: averaged_poll_2 ---


def averaged_poll_2(sensor: Some[DeflectionSensing], samples: Int) -> Float64:
    var total: Float64 = 0.0
    for _ in range(samples):
        total += sensor.fetch_reading()
    return total / Float64(samples)


def test_averaged_poll_2() raises:
    var sensor = CapacitiveSensor()
    assert_equal(averaged_poll_2(sensor, 4), 21.5)


# --- Named form for same-type pairs: compare_readings ---


def compare_readings[
    SensorType: DeflectionSensing
](a: SensorType, b: SensorType) -> Float64:
    return a.fetch_reading() - b.fetch_reading()


def test_compare_readings() raises:
    var a = CapacitiveSensor()
    var b = CapacitiveSensor()
    assert_equal(compare_readings(a, b), 0.0)


# --- Combining traits: Loggable, SensorLike, SmartSensor ---


trait Loggable:
    def log(self, message: String):
        ...


def poll_and_log[T: DeflectionSensing & Loggable](sensor: T):
    # print(sensor.fetch_reading())
    sensor.log("Polling sensor")


comptime SensorLike = DeflectionSensing & Loggable


@fieldwise_init
struct SmartSensor(Copyable, SensorLike):
    def fetch_reading(self) -> Float64:
        return 18.2

    def log(self, message: String):
        # print("reading logged")
        pass


def test_poll_and_log() raises:
    var smart = SmartSensor()
    assert_equal(smart.fetch_reading(), 18.2)
    poll_and_log(smart)


# --- Default implementations: DefaultLoggable / BasicSensor ---


trait DefaultLoggable:
    def log(self, message: String):
        # print("reading logged")
        pass


@fieldwise_init
struct BasicSensor(Copyable, DefaultLoggable):
    pass


def test_default_loggable() raises:
    var basic = BasicSensor()
    basic.log("test")  # reading logged (inherited default)


# --- Resolving a default-implementation conflict ---
# The doc shows the conflict as a compile error
# (struct Gateway(PowerCycle, Rebootable): pass). This gives Gateway
# its own restart() to demonstrate the resolution the prose describes.


trait PowerCycle:
    def restart(self):
        # print("Restarting via power cycle")
        pass


trait Rebootable:
    def restart(self):
        # print("Restarting via soft reboot")
        pass


@fieldwise_init
struct Gateway(PowerCycle, Rebootable):
    def restart(self):
        # print("Gateway restart resolved")
        pass


def test_gateway_restart_override() raises:
    var gw = Gateway()
    gw.restart()  # Gateway restart resolved


def main() raises:
    test_eddy_current_sensor()
    test_capacitive_sensor()
    test_timer_pausable()
    test_averaged_poll()
    test_averaged_poll_2()
    test_compare_readings()
    test_poll_and_log()
    test_default_loggable()
    test_gateway_restart_override()
