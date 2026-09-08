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

# RUN: %parse-mojo-isolated %s -verify-diagnostics -o /dev/null


# We should be able to infer something with a concrete origin even if it
# requires an upcast.
def takeA[origin: Origin, T: AnyType](ref[origin] a: String, b: T):
    pass


def infer_ref_argument():
    var s: String
    var t: String
    takeA(s, t)  # Ok.
    takeA[AnyOrigin[mut=True]](s, t)  # Ok.


@fieldwise_init
struct TwoIntParamStruct[a: Int, b: Int](Movable where False):
    pass


# expected-note @below {{function declared here}}
def take_two_int_dep[x: Int](a: TwoIntParamStruct[x, x + 1]):
    pass


# expected-note @+1 {{function declared here}}
def take_tied_values[
    first1: Int, first2: Int, second: Int
](a: TwoIntParamStruct[first1, second], b: TwoIntParamStruct[first2, second],):
    pass


def infer_two_param_dep_struct[y: Int]():
    take_two_int_dep(TwoIntParamStruct[1, 2]())
    # expected-error @+2 {{invalid call to 'take_two_int_dep': value passed to 'a' cannot be converted from 'TwoIntParamStruct[Int(2), Int(2)]' to 'TwoIntParamStruct[Int(2), Int(3)]'}}
    # expected-note @+1 {{.b of the first value is 'Int(2)' but the second value is 'Int(3)'}}
    take_two_int_dep(TwoIntParamStruct[2, 2]())
    take_two_int_dep(TwoIntParamStruct[b=2, a=1]())

    take_two_int_dep(TwoIntParamStruct[y, y + 1]())
    # expected-error @+2 {{invalid call to 'take_two_int_dep': value passed to 'a' cannot be converted from 'TwoIntParamStruct[y, (y + Int(2))]' to 'TwoIntParamStruct[y, (y + Int(1))]'}}
    # expected-note @+1 {{.b of the first value is '(y + Int(2))' but the second value is '(y + Int(1))'}}
    take_two_int_dep(TwoIntParamStruct[y, y + 2]())

    # expected-error @+2 {{invalid call to 'take_tied_values': value passed to 'b' cannot be converted from 'TwoIntParamStruct[Int(20), Int(2)]' to 'TwoIntParamStruct[Int(20), Int(1)]'}}
    # expected-note @+1 {{.b of the first value is 'Int(2)' but the second value is 'Int(1)'}}
    take_tied_values(TwoIntParamStruct[10, 1](), TwoIntParamStruct[20, 2]())
