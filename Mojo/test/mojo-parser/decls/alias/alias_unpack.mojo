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
# RUN: %parse-mojo-isolated %s | FileCheck %s

# TODO: to make `var : T1 = 1` work, we need to make tuple::__method__ parser
# foldable.

# works without parens
comptime _, (b, c) = 1, (2, 3.0)


# TODO(MOCO-2764)
# alias T1, (_, T3) = (Int, (Int, FloatDyn))


def use[T: AnyType](t: T):
    pass


def foo():
    # CHECK: kgen.param.constant: !Int
    # CHECK: kgen.param.constant: !FloatDyn
    use(b)
    use(c)

    # var x: T1
    # var y: T3
