# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Implements the source range struct.
"""


@value
@register_passable("trivial")
struct _SourceLocation(Stringable):
    var file_name: StringLiteral
    var function_name: StringLiteral
    var line: Int

    fn __str__(self) -> String:
        return (
            str(self.file_name)
            + ":"
            + str(self.function_name)
            + ":"
            + str(self.line)
        )
