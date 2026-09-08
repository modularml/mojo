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

# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s

# A struct that conforms to a trait but mismatches an inherited defaulted
# method's signature must report a diagnostic, not crash.

# CHECK: error: 'Map[mapFn]' does not implement all requirements for 'Strategy'

@fieldwise_init
struct Map[Strat: Strategy, Dest: Copyable, //, mapFn: def (Strat.Value) thin -> Dest](Strategy):
    var strat: Self.Strat

    comptime Value = Self.Dest

    def value(mut self) raises -> Self.Value:
        return Self.mapFn(self.strat.value())


trait Strategy(Deinitable, Movable):
    comptime Value: Copyable

    def value(mut self) raises -> Self.Value:
        ...

    def map[To: Copyable, //, mapper: def (Self.Value) thin -> To](var self) -> Map[mapper]:
        return Map[mapper](self^)
