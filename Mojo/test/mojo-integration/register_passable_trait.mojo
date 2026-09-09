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
# RUN: kgen -elaborate %s -S -o - | FileCheck %s --check-prefix=ELABORATE
# RUN: %mojo -debug-level full %s 3 | FileCheck %s

from std.sys import argv


trait RGTrivialTrait(TrivialRegisterPassable):
    def doSomething(self):
        ...


@fieldwise_init
struct Conforms(RGTrivialTrait):
    var x: Int

    @no_inline
    def doSomething(self):
        print(self.x)


# ELABORATE: kgen.func @"{{.*}}bar{{.*}}"(%arg0: !kgen.scalar<index>)
# ELABORATE-NEXT: kgen.call @"{{.*}}::Conforms::doSomething{{.*}}"(%arg0) : (!kgen.scalar<index>) -> ()
@no_inline
def bar[x: RGTrivialTrait](y: x):
    y.doSomething()


def main() raises:
    var t = Conforms(atol(argv()[1]))
    # CHECK: 3
    bar(t)
