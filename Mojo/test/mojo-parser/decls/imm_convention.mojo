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

# 'imm' is the new spelling of the immutable-borrow argument and
# closure-capture convention; 'read' is a removed synonym that warns with a
# fixit (fixit positions are covered by imm_convention_fixit.mojo, IR
# equivalence by imm_convention_ir.mojo).

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics %s

# 'imm' parses without diagnostics in argument and capture position.


def imm_arg(imm x: Int) -> Int:
    return x


def imm_capture() -> Int:
    var base = 0

    def closure() {imm base} -> Int:
        return base

    return closure()


# // -----

# 'read' still parses but is deprecated in argument position...

# expected-error @below {{'read' was removed; use 'imm'}}
def read_arg(read x: Int) -> Int:
    return x

# // -----

# ...and in capture position.


def read_capture() -> Int:
    var base = 0

    # expected-error @below {{'read' was removed; use 'imm'}}
    def closure() {read base} -> Int:
        return base

    return closure()


# // -----

# Like 'read', 'imm' is claimed by convention parsing in argument lists, so it
# cannot be an argument name.

# expected-error @below {{expected argument name}}
def imm_as_arg_name(imm: Int):
    pass

# // -----

# Parameter lists don't take conventions, so 'imm' remains a valid parameter
# name there, and it is still a soft keyword usable as a local variable name.


def param_named_imm[imm: Int]() -> Int:
    var imm2 = imm
    return imm2


def local_named_imm() -> Int:
    var imm = 5
    return imm
