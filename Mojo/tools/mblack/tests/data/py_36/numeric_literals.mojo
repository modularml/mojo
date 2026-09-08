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
#   Path:   tests/data/py_36/numeric_literals.py
#
# ===----------------------------------------------------------------------=== #

#!/usr/bin/env python3.6

x = 123456789
x = 123456
x = 0.1
x = 1.0
x = 1e1
x = 1e-1
x = 1.000_000_01
x = 123456789.123456789
x = 123456789.123456789e123456789
x = 123456789e123456789
x = 123456789j
x = 123456789.123456789j
x = 0xB1ACC
x = 0b1011
x = 0o777
x = 0.000000006
x = 10000
x = 133333

# output


#!/usr/bin/env python3.6

x = 123456789
x = 123456
x = 0.1
x = 1.0
x = 1e1
x = 1e-1
x = 1.000_000_01
x = 123456789.123456789
x = 123456789.123456789e123456789
x = 123456789e123456789
x = 123456789j
x = 123456789.123456789j
x = 0xB1ACC
x = 0b1011
x = 0o777
x = 0.000000006
x = 10000
x = 133333
