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
#   Path:   tests/data/simple_cases/fmtonoff2.py
#
# ===----------------------------------------------------------------------=== #

import pytest

TmSt = 1
TmEx = 2

# fmt: off

# Test data:
#   Position, Volume, State, TmSt/TmEx/None, [call, [arg1...]]

@pytest.mark.parametrize('test', [

    # Test don't manage the volume
    [
        ('stuff', 'in')
    ],
])
def test_fader(test):
    pass

def check_fader(test):

    pass

def verify_fader(test):
  # misaligned comment
    pass

def verify_fader(test):
    """Hey, ho."""
    assert test.passed()

def test_calculate_fades():
    calcs = [
        # one is zero/none
        (0, 4, 0, 0, 10,        0, 0, 6, 10),
        (None, 4, 0, 0, 10,     0, 0, 6, 10),
    ]

# fmt: on
