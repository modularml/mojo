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

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   tests/data/py_39/python39.py
#
# ===----------------------------------------------------------------------=== #

#!/usr/bin/env python3.9


@relaxed_decorator[0]
def f():
    ...


@relaxed_decorator[
    extremely_long_name_that_definitely_will_not_fit_on_one_line_of_standard_length
]
def f():
    ...


@extremely_long_variable_name_that_doesnt_fit := complex.expression(
    with_long="arguments_value_that_wont_fit_at_the_end_of_the_line"
)
def f():
    ...


# output


#!/usr/bin/env python3.9


@relaxed_decorator[0]
def f():
    ...


@relaxed_decorator[
    extremely_long_name_that_definitely_will_not_fit_on_one_line_of_standard_length
]
def f():
    ...


@extremely_long_variable_name_that_doesnt_fit := complex.expression(
    with_long="arguments_value_that_wont_fit_at_the_end_of_the_line"
)
def f():
    ...
