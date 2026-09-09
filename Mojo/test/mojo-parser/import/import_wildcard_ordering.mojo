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

# Wildcard imports resolve last to first, matching Python, where each
# `from x import *` rebinds names by assignment so the textually-last import
# wins collisions. Repeating an import repositions it: the trailing
# `wildcard_shadow_a` import moves it after `wildcard_shadow_b`, so `a` wins
# the collision on shadowed_fn even though `b` was written after it once.

# RUN: %parse-mojo-isolated -I=%S/inputs %s | FileCheck %s

from wildcard_shadow_a import *
from wildcard_shadow_b import *
from wildcard_shadow_a import *

# CHECK-LABEL: lit.fn @"main
def main():
    # CHECK: lit.call {{.*}}@wildcard_shadow_a::@"shadowed_fn
    _ = shadowed_fn()

    # Non-colliding names from the shadowed module still land.
    # CHECK: lit.call {{.*}}@wildcard_shadow_b::@"b_only_fn
    _ = b_only_fn()
