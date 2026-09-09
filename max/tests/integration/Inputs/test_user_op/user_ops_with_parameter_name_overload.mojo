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
from extensibility import *
from extensibility import InputTensor, OutputTensor
import extensibility


# This function has the same name as the parameter for the kernel registration.
def top_k():
    pass


@extensibility.register("parameter_name_overload")
struct ParameterNameOverload:
    # The top_k parameter matches the name of the top_k function defined above.
    @staticmethod
    def execute[
        top_k: Int
    ](result: OutputTensor, input: InputTensor,) -> None:
        print("Success!")
