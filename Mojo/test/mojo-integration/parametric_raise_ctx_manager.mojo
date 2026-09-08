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

# RUN: %mojo %s | FileCheck %s

##===----------------------------------------------------------------------===##
# Infer Error type in context manager.
##===----------------------------------------------------------------------===##


@fieldwise_init
struct ResourceError(ImplicitlyCopyable, Writable):
    var message: String

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ResourceError: ", self.message)


struct TypedResourceGuard(ImplicitlyCopyable):
    var name: String
    var suppress_errors: Bool

    def __init__(out self, name: String, suppress_errors: Bool = False):
        self.name = name
        self.suppress_errors = suppress_errors

    def __enter__(self) -> Self:
        print("Acquiring resource:", self.name)
        return self

    def __exit__(self):
        print("Releasing resource:", self.name, "(no error)")

    def __exit__[ErrType: AnyType](self, err: ErrType) -> Bool:
        print("Releasing resource:", self.name, "(typed error)")
        return self.suppress_errors


def use_resource_typed(name: String, should_fail: Bool) raises ResourceError:
    if should_fail:
        raise ResourceError("failed to use " + name)
    print("Successfully used resource:", name)


# CHECK:      Acquiring resource: network
# CHECK-NEXT: Working with network...
# CHECK-NEXT: Releasing resource: network (typed error)
# CHECK-NEXT: Continued after suppressed error
def main() raises:
    with TypedResourceGuard("network", suppress_errors=True):
        print("Working with network...")
        use_resource_typed("network", should_fail=True)
    print("Continued after suppressed error")
