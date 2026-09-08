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


# start-trait
trait SomeTrait:
    def required_method(self, x: Int):
        ...

    # end-trait


# start-conforming-struct
@fieldwise_init
struct SomeStruct(SomeTrait):
    def required_method(self, x: Int):
        print("hello traits", x)
        # end-conforming-struct


# start-generic-function
def fun_with_traits[T: SomeTrait](x: T):
    x.required_method(42)


def use_trait_function():
    var thing = SomeStruct()
    fun_with_traits(thing)
    # end-generic-function


def typed_collection():
    # start-typed-collection
    var my_list = List[Float64]()
    # end-typed-collection
    _ = my_list


def main():
    use_trait_function()
    typed_collection()
