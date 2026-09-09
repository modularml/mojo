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

# A module inside a package cannot reach a sibling module - or a sibling's
# symbols - by bare name without an explicit import. Siblings are unlisted
# children of the package, so a contained file's upward name lookup walks up
# into the package's (empty) scope and finds nothing; it never sees a sibling,
# nor __init__'s re-exports. The package's `__init__.mojo` deliberately
# re-exports `producer`, which makes `no_sibling_leak.producer` reachable from
# *outside* but must NOT make `producer` visible to its siblings.
#
# Note - diagnostics are emitted in the no_sibling_leak package: see there.
# RUN: %parse-mojo-isolated -verify-diagnostics -split-input-file -I=%S/inputs %s

from no_sibling_leak.consumer_module import consume

def main():
    _ = consume()

# // -----

# A sibling's SYMBOL that __init__ does NOT re-export is not visible without an
# import - it never leaked into the package scope, so this stays a hard error.
from no_sibling_leak.consumer_symbol import consume

def main():
    _ = consume()

# // -----

# A symbol that __init__ DOES re-export is reachable from a sibling only via the
# same temporary deprecation warning (the package used to wildcard-import
# __init__ into its scope). It too becomes a hard error in a future release.
from no_sibling_leak.consumer_reexport import consume

def main():
    _ = consume()
