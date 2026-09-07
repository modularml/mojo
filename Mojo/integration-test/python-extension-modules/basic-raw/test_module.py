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

# Imports from 'mojo_module.so'
import mojo_module  # type: ignore[import-not-found]


def test_basic_raw() -> None:
    result = mojo_module.mojo_count_args(1, 2)
    assert result == 2

    result = mojo_module.mojo_count_args_with_kwargs(1, 2, 3)
    assert result == 3

    result = mojo_module.mojo_count_args_with_kwargs(1, 2, c=3)
    assert result == 3

    result = mojo_module.mojo_count_args_with_kwargs(a=1, b=2, c=3)
    assert result == 3


def test_def_py_c_method() -> None:
    # Test non-keyword arg variant (PyCFunction)
    counter = mojo_module.TestCounter()
    assert counter.count_args(1, 2) == 2

    # Test keyword arg variant (PyCFunctionWithKeywords)
    counter2 = mojo_module.TestCounter()
    assert counter2.count_args_with_kwargs(1, 2, 3) == 3

    # Test static methods
    assert mojo_module.TestCounter.static_count_args(1, 2) == 2
    assert mojo_module.TestCounter.static_count_args_with_kwargs(1, 2, 3) == 3
