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

from .base import MyStruct


# User will import this directly, not the extension.
struct IntConfig:
    def __init__(out self):
        pass


# This extension will hitch a ride on any imports of anything in this file
# (like IntConfig) to make itself known to anybody who deals with anyone in
# this file.
__extension MyStruct:
    def intermediate_method(self):
        print("intermediate_method from extension")
