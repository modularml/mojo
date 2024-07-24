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
# RUN: %mojo %s


def main():
    test_type_from_none()


struct FromNone:
    var value: Int

    fn __init__(inout self, none: NoneType):
        self.value = -1

    fn __init__(inout self, value: Int):
        self.value = value


def test_type_from_none():
    obj = FromNone(5)

    obj = FromNone(None)

    # -------------------------------------
    # Test implicit conversion from `None`
    # -------------------------------------

    fn foo(arg: FromNone):
        pass

    # FIXME:
    #   This currently fails, because it requires 2 "hops" of conversion:
    #       1. !kgen.none => NoneType
    #       2. NoneType => FromNone
    # foo(None)
    #
    #   But, interestingly, this does not fail?
    var obj2: FromNone = None
