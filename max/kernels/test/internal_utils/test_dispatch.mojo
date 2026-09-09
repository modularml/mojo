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

from internal_utils import Table, TuningConfig


struct TestConfig(TrivialRegisterPassable, TuningConfig):
    var m: Int
    var n: Int
    var k: Int
    var tile: Int

    def __init__(out self, m: Int, n: Int, k: Int, tile: Int):
        self.m = m
        self.n = n
        self.k = k
        self.tile = tile

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "config: ",
            "m:",
            self.m,
            "/n:",
            self.n,
            "/k:",
            self.k,
        )


comptime TEST_TABLE = Table(
    [
        TestConfig(m=1, n=128, k=256, tile=16),
        TestConfig(m=2, n=128, k=256, tile=32),
        TestConfig(m=16, n=128, k=256, tile=64),
        TestConfig(m=1, n=256, k=256, tile=128),
    ],
    "test",
)


def dispatch[static_n: Int, static_k: Int](m: Int) raises -> Int:
    comptime nk_indices = TEST_TABLE.query_index(
        rule=lambda (config: TestConfig) -> Bool: config.n == static_n
        and config.k == static_k
    )

    comptime m_values = TEST_TABLE.query_values[Int, domain=nk_indices](
        rule=lambda (config: TestConfig) -> Int: config.m
    )
    comptime assert len(m_values) == 3
    comptime assert m_values[0] == 1
    comptime assert m_values[1] == 2
    comptime assert m_values[2] == 16

    comptime for static_m in m_values:
        if m <= static_m:
            comptime indices = TEST_TABLE.query_index[domain=nk_indices](
                rule=lambda (config: TestConfig) -> Bool: config.m == static_m
            )
            comptime assert len(indices) == 1
            comptime entry = TEST_TABLE.configs[indices[0]]
            return entry.tile

    return -1


def main() raises:
    assert dispatch[128, 256](1) == 16
    assert dispatch[128, 256](2) == 32
    assert dispatch[128, 256](3) == 64
    assert dispatch[128, 256](17) == -1

    comptime matching = TEST_TABLE.find(
        rule=lambda (config: TestConfig) -> Bool: config.tile == 128
    )
    comptime assert len(matching) == 1
