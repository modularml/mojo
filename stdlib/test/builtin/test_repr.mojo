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


@value
struct Dog(Representable):
    var name: String
    var age: Int

    fn __repr__(self) -> String:
        return "Dog(name=" + repr(self.name) + ", age=" + repr(self.age) + ")"


def test_explicit_conformance():
    dog = Dog(name="Fido", age=3)
    assert_equal(repr(dog), "Dog(name='Fido', age=3)")


@value
struct Cat:
    var name: String
    var age: Int

    fn __repr__(self) -> String:
        return "Cat(name=" + repr(self.name) + ", age=" + repr(self.age) + ")"


def test_implicit_conformance():
    cat = Cat(name="Whiskers", age=2)
    assert_equal(repr(cat), "Cat(name='Whiskers', age=2)")


def test_none_representation():
    assert_equal(repr(None), "None")


def main():
    test_explicit_conformance()
    test_implicit_conformance()
    test_none_representation()
