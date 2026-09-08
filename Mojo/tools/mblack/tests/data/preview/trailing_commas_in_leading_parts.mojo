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
#   Path:   tests/data/preview/trailing_commas_in_leading_parts.py
#
# ===----------------------------------------------------------------------=== #

zero(one,).two(three,).four(
    five,
)

func1(arg1).func2(arg2,).func3(arg3).func4(
    arg4,
).func5(arg5)

# Inner one-element tuple shouldn't explode
func1(arg1).func2(arg1, (one_tuple,)).func3(arg3)

(a, b, c, d,) = func1(
    arg1
) and func2(arg2)


# Example from https://github.com/psf/black/issues/3229
def refresh_token(self, device_family, refresh_token, api_key):
    return self.orchestration.refresh_token(
        data={
            "refreshToken": refresh_token,
        },
        api_key=api_key,
    )["extensions"]["sdk"]["token"]


# Edge case where a bug in a working-in-progress version of
# https://github.com/psf/black/pull/3370 causes an infinite recursion.
assert (
    long_module.long_class.long_func().another_func()
    == long_module.long_class.long_func()["some_key"].another_func(arg1)
)


# output


zero(
    one,
).two(
    three,
).four(
    five,
)

func1(arg1).func2(
    arg2,
).func3(arg3).func4(
    arg4,
).func5(arg5)

# Inner one-element tuple shouldn't explode
func1(arg1).func2(arg1, (one_tuple,)).func3(arg3)

(
    a,
    b,
    c,
    d,
) = func1(
    arg1
) and func2(arg2)


# Example from https://github.com/psf/black/issues/3229
def refresh_token(self, device_family, refresh_token, api_key):
    return self.orchestration.refresh_token(
        data={
            "refreshToken": refresh_token,
        },
        api_key=api_key,
    )["extensions"]["sdk"]["token"]


# Edge case where a bug in a working-in-progress version of
# https://github.com/psf/black/pull/3370 causes an infinite recursion.
assert (
    long_module.long_class.long_func().another_func()
    == long_module.long_class.long_func()["some_key"].another_func(arg1)
)
