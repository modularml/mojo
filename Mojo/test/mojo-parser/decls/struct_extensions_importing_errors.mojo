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

# RUN: %parse-mojo-isolated --mojo-disable-builtins -I %S/inputs %s -mlir-print-debuginfo -split-input-file -verify-diagnostics

# @expected-note @below {{conflicts with this previous declaration}}
from struct_package_for_conflict import PlainStruct


# @expected-error @below {{cannot define a struct here with name 'PlainStruct'}}
struct PlainStruct:
    pass


__extension PlainStruct:
    def sparklebark(self: PlainStruct):
        pass
