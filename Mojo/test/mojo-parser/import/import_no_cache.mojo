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

# RUN: %parse-mojo-isolated -I=%S/inputs %s | FileCheck %s

from test_package.module import *


# CHECK-LABEL: lit.fn @"foo
def foo():
    var x = Wrapper(33)
    var y = x.data


# Even though ParameterizedType is referenced in an alias in Wrapper, the alias
# itself is unused by this file. ParameterizedType should have been removed as
# an unreachable decl.
# CHECK-NOT: lit.struct.decl @ParameterizedType
