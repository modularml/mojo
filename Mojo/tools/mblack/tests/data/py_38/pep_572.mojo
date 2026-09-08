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
#   Path:   tests/data/py_38/pep_572.py
#
# ===----------------------------------------------------------------------=== #

(a := 1)
(a := a)
if (match := pattern.search(data)) is None:
    pass
if match := pattern.search(data):
    pass
[y := f(x), y**2, y**3]
filtered_data = [y for x in data if (y := f(x)) is None]
(y := f(x))
y0 = (y1 := f(x))
foo(x=(y := f(x)))


def foo(answer=(p := 42)):
    pass


def foo(answer: (p := 42) = 5):
    pass


lambda: (x := 1)
(x := lambda: 1)
(x := lambda: (y := 1))
lambda line: (m := re.match(pattern, line)) and m.group(1)
x = (y := 0)
(z := (y := (x := 0)))
(info := (name, phone, *rest))
(x := 1, 2)
(total := total + tax)
len(lines := f.readlines())
foo(x := 3, cat="vector")
foo(cat=(category := "vector"))
if any(len(longline := l) >= 100 for l in lines):
    print(longline)
if env_base := os.environ.get("PYTHONUSERBASE", None):
    return env_base
if self._is_special and (ans := self._check_nans(context=context)):
    return ans
foo(b := 2, a=1)
foo((b := 2), a=1)
foo(c=(b := 2), a=1)

while x := f(x):
    pass
while x := f(x):
    pass
