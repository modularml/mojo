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

# RUN: %parse-mojo-isolated --verify-diagnostics %s


@fieldwise_init
struct Foo(Movable where False):
    def __getitem_param__[T: AnyType](self) raises:
        pass


# expected-note @below {{or mark surrounding function as 'raises'}}
def main():
    var f = Foo()

    # expected-error @below {{cannot call function that may raise in a context that cannot raise}}
    # expected-note @below {{try surrounding the call in a 'try' block}}
    _ = f[Int]
