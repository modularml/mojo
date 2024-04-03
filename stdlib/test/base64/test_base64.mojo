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
# RUN: %mojo -debug-level full %s | FileCheck %s

from base64 import b64encode


# CHECK-LABEL: test_b64encode
fn test_b64encode():
    print("== test_b64encode")

    # CHECK: YQ==
    print(b64encode("a"))

    # CHECK: Zm8=
    print(b64encode("fo"))

    # CHECK: SGVsbG8gTW9qbyEhIQ==
    print(b64encode("Hello Mojo!!!"))

    # CHECK: SGVsbG8g8J+UpSEhIQ==
    print(b64encode("Hello 🔥!!!"))

    # CHECK: dGhlIHF1aWNrIGJyb3duIGZveCBqdW1wcyBvdmVyIHRoZSBsYXp5IGRvZw==
    print(b64encode("the quick brown fox jumps over the lazy dog"))

    # CHECK: QUJDREVGYWJjZGVm
    print(b64encode("ABCDEFabcdef"))


fn main():
    test_b64encode()
