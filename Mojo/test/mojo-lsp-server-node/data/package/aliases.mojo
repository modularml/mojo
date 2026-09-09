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

from std.ffi import RTLD

comptime IntAlias = 12
"""Int alias summary

Int alias description."""
comptime ExplicitIntAlias: Int = 123


def function() -> Int:
    comptime AliasInsideFunction = "sdfsdf"


comptime AliasToAlias = IntAlias


struct StructWithAlias:
    comptime AliasInStruct = Int


comptime AliasInStructRef = StructWithAlias.AliasInStruct

comptime ExternalAlias = RTLD.LAZY
