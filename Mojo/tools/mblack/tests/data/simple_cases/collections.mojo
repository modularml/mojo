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
#   Path:   tests/data/simple_cases/collections.py
#
# ===----------------------------------------------------------------------=== #

import core, time, a

from . import A, B, C

# keeps existing trailing comma
from foo import (
    bar,
)

# also keeps existing structure
from foo import (
    baz,
    qux,
)

# `as` works as well
from foo import (
    xyzzy as magic,
)

a = {
    1,
    2,
    3,
}
b = {1, 2, 3}
c = {
    1,
    2,
    3,
}
x = (1,)
y = (narf(),)
nested = {
    (1, 2, 3),
    (4, 5, 6),
}
nested_no_trailing_comma = {(1, 2, 3), (4, 5, 6)}
nested_long_lines = [
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "cccccccccccccccccccccccccccccccccccccccc",
    (1, 2, 3),
    "dddddddddddddddddddddddddddddddddddddddd",
]
{
    "oneple": (1,),
}
{"oneple": (1,)}
["ls", "lsoneple/%s" % (foo,)]
x = {"oneple": (1,)}
y = {
    "oneple": (1,),
}
assert False, (
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    " wraps %s" % bar
)

# looping over a 1-tuple should also not get wrapped
for x in (1,):
    pass
for (x,) in (1,), (2,), (3,):
    pass

[
    1,
    2,
    3,
]

division_result_tuple = (6 / 2,)
print("foo %r", (foo.bar,))

if True:
    IGNORED_TYPES_FOR_ATTRIBUTE_CHECKING = (
        Config.IGNORED_TYPES_FOR_ATTRIBUTE_CHECKING
        | {pylons.controllers.WSGIController}
    )

if True:
    ec2client.get_waiter("instance_stopped").wait(
        InstanceIds=[instance.id],
        WaiterConfig={
            "Delay": 5,
        },
    )
    ec2client.get_waiter("instance_stopped").wait(
        InstanceIds=[instance.id],
        WaiterConfig={
            "Delay": 5,
        },
    )
    ec2client.get_waiter("instance_stopped").wait(
        InstanceIds=[instance.id],
        WaiterConfig={
            "Delay": 5,
        },
    )

# output


import core, time, a

from . import A, B, C

# keeps existing trailing comma
from foo import (
    bar,
)

# also keeps existing structure
from foo import (
    baz,
    qux,
)

# `as` works as well
from foo import (
    xyzzy as magic,
)

a = {
    1,
    2,
    3,
}
b = {1, 2, 3}
c = {
    1,
    2,
    3,
}
x = (1,)
y = (narf(),)
nested = {
    (1, 2, 3),
    (4, 5, 6),
}
nested_no_trailing_comma = {(1, 2, 3), (4, 5, 6)}
nested_long_lines = [
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "cccccccccccccccccccccccccccccccccccccccc",
    (1, 2, 3),
    "dddddddddddddddddddddddddddddddddddddddd",
]
{
    "oneple": (1,),
}
{"oneple": (1,)}
["ls", "lsoneple/%s" % (foo,)]
x = {"oneple": (1,)}
y = {
    "oneple": (1,),
}
assert False, (
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    " wraps %s" % bar
)

# looping over a 1-tuple should also not get wrapped
for x in (1,):
    pass
for (x,) in (1,), (2,), (3,):
    pass

[
    1,
    2,
    3,
]

division_result_tuple = (6 / 2,)
print("foo %r", (foo.bar,))

if True:
    IGNORED_TYPES_FOR_ATTRIBUTE_CHECKING = (
        Config.IGNORED_TYPES_FOR_ATTRIBUTE_CHECKING
        | {pylons.controllers.WSGIController}
    )

if True:
    ec2client.get_waiter("instance_stopped").wait(
        InstanceIds=[instance.id],
        WaiterConfig={
            "Delay": 5,
        },
    )
    ec2client.get_waiter("instance_stopped").wait(
        InstanceIds=[instance.id],
        WaiterConfig={
            "Delay": 5,
        },
    )
    ec2client.get_waiter("instance_stopped").wait(
        InstanceIds=[instance.id],
        WaiterConfig={
            "Delay": 5,
        },
    )
