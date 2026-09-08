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
#   Path:   tests/data/py_36/numeric_literals_skip_underscores.py
#
# ===----------------------------------------------------------------------=== #

#!/usr/bin/env python3.6

x = 123456789
x = 1_2_3_4_5_6_7
x = 1e1
x = 0xB1ACC
x = 0.00_00_006
x = 12_34_567j
x = 0.1_2
x = 1_2.0

# output

#!/usr/bin/env python3.6

x = 123456789
x = 1_2_3_4_5_6_7
x = 1e1
x = 0xB1ACC
x = 0.00_00_006
x = 12_34_567j
x = 0.1_2
x = 1_2.0
