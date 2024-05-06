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


trait DivModable(CollectionElement):
    fn __divmod__(self: Self, other: Self) -> Tuple[Self, Self]:
        ...


fn divmod[T: DivModable](a: T, b: T) -> Tuple[T, T]:
    """Performs integer division and returns the quotient and the remainder.

    Currently supported only for integers. Support for more standard library types
    like Int8, Int16... is planned.

    This method calls `a.__divmod__(b)`, thus, the actual implementation of
    divmod should go in the `__divmod__` method of the struct of `a`.

    Args:
        a: The dividend.
        b: The divisor.

    Returns:
        A `Tuple` containing the quotient and the remainder.
    """
    return a.__divmod__(b)
