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
# RUN: env MODULAR_PROFILE_FILENAME="-" %mojo-no-debug %s | FileCheck %s


from std.os import abort

from std.runtime._asyncrt import create_task
from max.runtime.tracing import Trace, TraceLevel


def test_tracing[level: TraceLevel, enabled: Bool]() raises:
    @__parameter
    async def test_tracing_add[enabled: Bool, lhs: Int](rhs: Int) -> Int:
        comptime s1 = "ENABLED: trace event 2" if enabled else StaticString(
            "DISABLED: trace event 2"
        )
        comptime s2 = "ENABLED: detail event 2" if enabled else String(
            "DISABLED: detail event 2"
        )
        try:
            with Trace[level](s1, s2):
                return lhs + rhs
        except e:
            abort(String(e))

    @__parameter
    async def test_tracing_add_two_of_them[
        enabled: Bool
    ](a: Int, b: Int) -> Int:
        var t0 = create_task(test_tracing_add[enabled, 1](a))
        var t1 = create_task(test_tracing_add[enabled, 2](b))
        return await t0 + await t1

    comptime s1 = "ENABLED: trace event 1" if enabled else StaticString(
        "DISABLED: trace event 1"
    )
    comptime s2 = "ENABLED: detail event 1" if enabled else String(
        "DISABLED: detail event 1"
    )
    with Trace[level](s1, s2):
        var task = create_task(test_tracing_add_two_of_them[enabled](10, 20))
        _ = task.wait()


def main() raises:
    # CHECK-LABEL: test_tracing_enabled
    print("== test_tracing_enabled")
    test_tracing[TraceLevel.ALWAYS, True]()
    # CHECK: "ENABLED: trace event 1"
    # CHECK-SAME: "ENABLED: detail event 1"
    # CHECK-SAME: "ENABLED: trace event 2"
    # CHECK-SAME: "ENABLED: detail event 2"
