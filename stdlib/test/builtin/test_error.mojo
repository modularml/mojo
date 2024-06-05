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

from testing import assert_equal


def raise_an_error():
    raise Error("MojoError: This is an error!")


def test_error_raising():
    try:
        raise_an_error()
    except e:
        assert_equal(str(e), "MojoError: This is an error!")


def test_from_and_to_string():
    var myString: String = "FOO"
    var error = Error(myString)
    assert_equal(str(error), "FOO")

    assert_equal(str(Error("bad")), "bad")
    assert_equal(repr(Error("err")), "Error('err')")


def main():
    test_error_raising()
    test_from_and_to_string()
