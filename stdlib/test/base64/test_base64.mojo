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

from base64 import b64encode, b64decode
from testing import assert_equal


def test_b64encode():
    assert_equal(b64encode("a"), "YQ==")

    assert_equal(b64encode("fo"), "Zm8=")

    assert_equal(b64encode("Hello Mojo!!!"), "SGVsbG8gTW9qbyEhIQ==")

    assert_equal(b64encode("Hello 🔥!!!"), "SGVsbG8g8J+UpSEhIQ==")

    assert_equal(
        b64encode("the quick brown fox jumps over the lazy dog"),
        "dGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZw==",
    )

    assert_equal(b64encode("ABCDEFabcdef"), "QUJDREVGYWJjZGVm")


def test_b64decode():
    assert_equal(b64decode("YQ=="), "a")

    assert_equal(b64decode("Zm8="), "fo")

    assert_equal(b64decode("SGVsbG8gTW9qbyEhIQ=="), "Hello Mojo!!!")

    assert_equal(b64decode("SGVsbG8g8J+UpSEhIQ=="), "Hello 🔥!!!")

    assert_equal(
        b64decode(
            "dGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZw=="
        ),
        "the quick brown fox jumps over the lazy dog",
    )

    assert_equal(b64decode("QUJDREVGYWJjZGVm"), "ABCDEFabcdef")


def main():
    test_b64encode()
    test_b64decode()
