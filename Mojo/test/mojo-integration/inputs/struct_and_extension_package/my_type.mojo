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


struct ZInt:
    def __init__(out self):
        pass


# Struct with a basic constructor
struct MyType:
    var value: ZInt

    def __init__(out self):
        self.value = ZInt()


# Extension that adds an alternate constructor
__extension MyType:
    def __init__(out self, initial_value: ZInt):
        self.value = ZInt()
