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

# RUN: %parse-mojo-isolated -I=%S/inputs %s | FileCheck %s

# An explicit import from a package must make *all* extensions re-exported by
# the package's wildcard imports visible, not just those from the wildcard
# that happened to resolve first. Lazy lookups stop draining a scope's
# wildcards as soon as one provides the requested name, which leaves the
# aggregate "extension:" entry partially populated:
# - `wc_ext_pkg` / `wc_ext_pkg_rev`: the imported name is provided by one of
#   the wildcards, so the String extension sits before or after the providing
#   wildcard; each order is the blind spot of one drain direction.
# - `wc_ext_pkg_direct`: the imported name is declared directly in __init__,
#   so the "extension:" lookup itself does the draining and would stop at
#   whichever extension-carrying wildcard it reaches first, in either
#   direction.
from wc_ext_pkg import Widget
from wc_ext_pkg_rev import Gizmo
from wc_ext_pkg_direct import Direct

# CHECK-LABEL: lit.fn @"main
def main():
    var w = Widget()
    # CHECK: lit.call {{.*}}widget_extended
    _ = w.widget_extended()
    var g = Gizmo()
    # CHECK: lit.call {{.*}}gizmo_extended
    _ = g.gizmo_extended()
    var s = String()
    # CHECK: lit.call {{.*}}wc_ext_marker
    _ = s.wc_ext_marker()
    # CHECK: lit.call {{.*}}wc_ext_marker_rev
    _ = s.wc_ext_marker_rev()
    var d = Direct()
    # CHECK: lit.call {{.*}}wc_ext_marker_c
    _ = s.wc_ext_marker_c()
    # CHECK: lit.call {{.*}}wc_ext_marker_d
    _ = s.wc_ext_marker_d()
