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
#   Path:   tests/data/miscellaneous/python2_detection.py
#
# ===----------------------------------------------------------------------=== #

# This uses a similar construction to the decorators.py test data file FYI.

print "hello, world!"

###

exec "print('hello, world!')"

###

def set_position((x, y), value):
    pass

###

try:
    pass
except Exception, err:
    pass

###

raise RuntimeError, "I feel like crashing today :p"

###

`wow_these_really_did_exist`

###

10L

###

10l

###

0123

# output

print("hello python three!")

###

exec("I'm not sure if you can use exec like this but that's not important here!")

###

try:
    pass
except make_exception(1, 2):
    pass

###

try:
    pass
except Exception as err:
    pass

###

raise RuntimeError(make_msg(1, 2))

###

raise RuntimeError("boom!",)

###

def set_position(x, y, value):
    pass

###

10

###

0

###

000

###

0o12