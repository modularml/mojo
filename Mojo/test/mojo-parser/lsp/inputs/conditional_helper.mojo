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

# Generic struct with a conditional `Deinitable` conformance, used to
# check that destructor discharge works when this struct is imported and left
# signature-resolved only.


struct ConditionalHelper[T: Movable](
    Deinitable where conforms_to(T, Deinitable),
    Movable,
):
    var value: Self.T

    def __init__(out self, var value: Self.T):
        self.value = value^

    def __deinit__(deinit self) where conforms_to(Self.T, Deinitable):
        pass
