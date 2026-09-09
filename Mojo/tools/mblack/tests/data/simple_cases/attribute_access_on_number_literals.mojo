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
#   Path:   tests/data/simple_cases/attribute_access_on_number_literals.py
#
# ===----------------------------------------------------------------------=== #

x = (123456789).bit_count()
x = (123456).__abs__()
x = (0.1).is_integer()
x = (1.0).imag
x = (1e1).imag
x = (1e-1).real
x = (123456789.123456789).hex()
x = (123456789.123456789e123456789).real
x = (123456789e123456789).conjugate()
x = 123456789j.real
x = 123456789.123456789j.__add__(0b1011.bit_length())
x = 0xB1ACC.conjugate()
x = 0b1011.conjugate()
x = 0o777.real
x = (0.000000006).hex()
x = -100.0000j

if (10).real:
    ...

y = 100[no]
y = 100(no)

# output

x = (123456789).bit_count()
x = (123456).__abs__()
x = (0.1).is_integer()
x = (1.0).imag
x = (1e1).imag
x = (1e-1).real
x = (123456789.123456789).hex()
x = (123456789.123456789e123456789).real
x = (123456789e123456789).conjugate()
x = 123456789j.real
x = 123456789.123456789j.__add__(0b1011.bit_length())
x = 0xB1ACC.conjugate()
x = 0b1011.conjugate()
x = 0o777.real
x = (0.000000006).hex()
x = -100.0000j

if (10).real:
    ...

y = 100[no]
y = 100(no)
