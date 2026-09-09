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
#   Path:   tests/data/py_38/pep_570.py
#
# ===----------------------------------------------------------------------=== #


def positional_only_arg(a, /):
    pass


def all_markers(a, b, /, c, d, *, e, f):
    pass


def all_markers_with_args_and_kwargs(
    a_long_one,
    b_long_one,
    /,
    c_long_one,
    d_long_one,
    *args,
    e_long_one,
    f_long_one,
    **kwargs,
):
    pass


def all_markers_with_defaults(a, b=1, /, c=2, d=3, *, e=4, f=5):
    pass


def long_one_with_long_parameter_names(
    but_all_of_them,
    are_positional_only,
    arguments_mmmmkay,
    so_this_is_only_valid_after,
    three_point_eight,
    /,
):
    pass


lambda a, /: a

lambda a, b, /, c, d, *, e, f: a

lambda a, b, /, c, d, *args, e, f, **kwargs: args

lambda a, b=1, /, c=2, d=3, *, e=4, f=5: 1
