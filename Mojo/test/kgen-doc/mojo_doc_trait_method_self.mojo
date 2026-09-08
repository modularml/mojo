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
# RUN: kgen-doc %s | FileCheck %s

# Regression test: a trait method's implicit `self` must render as `self`, not
# as the internal `_Self` placeholder. A trait method is parented by the trait
# (not a struct), so the signature printer has to recognize a trait parent as a
# method context to substitute `self` (and suppress the spurious trailing `/`).

"""Module docstring."""


trait MyTrait:
    """A trait."""

    # CHECK: "signature": "def no_args(self)"
    def no_args(self):
        """A method with no explicit arguments."""
        ...

    # CHECK: "signature": "def with_arg(self, x: Int) -> Int"
    def with_arg(self, x: Int) -> Int:
        """A method with an argument and a result."""
        ...

    # CHECK: "signature": "def mut_self(mut self)"
    def mut_self(mut self):
        """A method taking `self` mutably."""
        ...

    # Static methods: `Self` still substitutes in the result type.
    # CHECK: "signature": "def create() -> Self"
    @staticmethod
    def create() -> Self:
        """A static method returning `Self`."""
        ...

    # CHECK: "signature": "def from_int(x: Int) -> Self"
    @staticmethod
    def from_int(x: Int) -> Self:
        """A static method with a normal first argument."""
        ...

    # TODO(compiler): sub-optimal print. The first argument of a static method
    # is incorrectly treated as `self` (the signature printer marks argument 0
    # as self whenever a self type is available, regardless of `@staticmethod`),
    # so a `Self`-typed first argument has its type elided: this renders as
    # `def combine(other) -> Self` instead of `def combine(other: Self) -> Self`.
    # This is a pre-existing quirk shared with struct static methods (see the
    # `isSelf` TODO in SignatureModel.cpp), not specific to traits.
    # CHECK: "signature": "def combine(other) -> Self"
    @staticmethod
    def combine(other: Self) -> Self:
        """A static method whose first argument is `Self`-typed."""
        ...
