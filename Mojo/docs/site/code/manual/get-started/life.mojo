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
from std.random import seed
from grid import Grid
from std.time import sleep


def main():
    seed()

    var grid = Grid[60, 20].random_grid(240)
    var gen = 1

    for _ in range(300):
        print(t"\033[H\033J\nGeneration: {gen}")
        grid.evolve()
        grid.print_grid()
        sleep(0.1)
        gen += 1
