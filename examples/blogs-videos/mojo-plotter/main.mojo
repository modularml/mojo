# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
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

from python import Python


fn main() raises:
    let torch = Python.import_module("torch")
    let x = torch.linspace(0, 10, 100)
    let y = torch.sin(x)
    plot(x, y)


def plot(x: PythonObject, y: PythonObject) -> None:
    let plt = Python.import_module("matplotlib.pyplot")
    plt.plot(x.numpy(), y.numpy())
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Plot of y = sin(x)")
    plt.grid(True)
    plt.show()
