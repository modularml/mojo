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

from test_package.module import dont_inline_me


# Prevent this function from being inlined into modules importing it. We wish to
# test that even when its definition exists in a separate package module (and
# that it, in turn, calls a function defined in another separate package), the
# module importing this function can invoke it and its dependent symbols.
@no_inline
def dont_inline_me_either():
    dont_inline_me()
