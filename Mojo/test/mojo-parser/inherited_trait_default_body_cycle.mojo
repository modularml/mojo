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

# Signature-resolving a trait method inherited from a parent trait must not
# resolve the parent default's body. The body here reaches a type that conforms
# to the inheriting trait, so resolving it re-enters the in-flight `Child.m` via
# that type's conformance check and used to report `attempt to resolve a
# recursive reference to declaration 'm'`. Only the parent's signature is needed
# to clone the inherited stub; the body resolves later, on demand, through the
# conforming type's wrapper.

# RUN: %parse-mojo-isolated %s | FileCheck %s


trait Parent:
    def m(self) -> Bool:
        var reached = Reached()
        return reached.n()


trait Child(Parent):
    def n(self) -> Bool:
        return True


struct Reached(Child):
    def __init__(out self):
        pass

    def m(self) -> Bool:
        return True


struct Driver(Child):
    def __init__(out self):
        pass

    def m(self) -> Bool:
        return True


# CHECK-LABEL: lit.fn @"main
def main():
    var driver = Driver()
    _ = driver.n()
