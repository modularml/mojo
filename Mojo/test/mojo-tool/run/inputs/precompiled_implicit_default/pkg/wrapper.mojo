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

# The implicit constructor is referenced only by the default value of
# `Container`'s `W` parameter, so a consumer that imports `Container` never
# materializes the constructor's declaration.


struct Wrapper(Copyable, Movable):
    var value: Int

    @implicit
    def __init__(out self, value: Int):
        self.value = value
