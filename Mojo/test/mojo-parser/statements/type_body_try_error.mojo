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

# Test that 'try' statement in type bodies emits an error.
# This is in a separate file because after 'try' is rejected, the 'except'
# keyword becomes an invalid token, causing a cascading error.

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# try in struct body
##===----------------------------------------------------------------------===##

struct StructWithTry(Movable where False):
    var x: Int
    # expected-error @below {{'try' must be contained in a function}}
    try:
        pass
    # expected-error @below {{unexpected token in expression}}
    except:
        pass

##===----------------------------------------------------------------------===##
# try in trait body
##===----------------------------------------------------------------------===##

trait TraitWithTry:
    # expected-error @below {{'try' must be contained in a function}}
    try:
        pass
    # expected-error @below {{unexpected token in expression}}
    except:
        pass

##===----------------------------------------------------------------------===##
# try in extension body
##===----------------------------------------------------------------------===##

struct ExtendedStructWithTry(Movable where False):
    var x: Int

__extension ExtendedStructWithTry:
    # expected-error @below {{'try' must be contained in a function}}
    try:
        pass
    # expected-error @below {{unexpected token in expression}}
    except:
        pass
