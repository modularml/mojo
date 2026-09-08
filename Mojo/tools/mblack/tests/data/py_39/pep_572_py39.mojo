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
#   Path:   tests/data/py_39/pep_572_py39.py
#
# ===----------------------------------------------------------------------=== #

# Unparenthesized walruses are now allowed in set literals & set comprehensions
# since Python 3.9
{x := 1, 2, 3}
{x4 := x**5 for x in range(7)}
# We better not remove the parentheses here (since it's a 3.10 feature)
x[(a := 1)]
x[(a := 1), (b := 3)]
