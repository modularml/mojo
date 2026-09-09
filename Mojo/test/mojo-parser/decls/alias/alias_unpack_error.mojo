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

# RUN: %parse-mojo-isolated -import-mojo -verify-diagnostics -split-input-file %s

# expected-error @below {{cannot unpack value of 'Tuple[Int, FloatDyn]' of 2 elements into 3 values}}
comptime a, (b, c, d) = (1, (2, 3.0))

# // -----

# expected-error @below {{invalid comptime declaration: expected an identifier or '_'}}
comptime t, True, c = 1, 2, 3


# // -----


struct A:
    # expected-error @below {{'comptime' constants inside structs must be declared separately; break this into individual declarations}}
    comptime a, b = 1, 2


# // -----


trait A:
    # expected-error @below {{a trait's associated types must be declared separately; break this into individual declarations}}
    comptime a, b = 1, 2


# // -----


# expected-note @below {{previous definition here}}
comptime a, b = 1, 2
# expected-error @below {{invalid redefinition of 'b'}}
comptime b, c = 2, 3


# // -----

def example[list: List[Int]]():
    # expected-error @below {{cannot destructure into list patterns yet}}
    comptime for [a, b] in list:
      pass
