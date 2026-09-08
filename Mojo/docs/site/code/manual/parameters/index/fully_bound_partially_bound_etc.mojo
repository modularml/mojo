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


# start-has-default
@fieldwise_init
struct HasDefault[x: Int, y: Int = 0]:
    pass


comptime UseDefault = HasDefault[10]
# end-has-default


@fieldwise_init
struct MyType[s: String, i: Int, i2: Int, b: Bool = True]:
    pass


# Fully-bound
def my_fn1(mt: MyType["Hello", 3, 4, True]):
    pass


# Partially-bound
def my_fn2(mt: MyType["Hola", _, _, True]):
    pass


# Unbound
def my_fn3(mt: MyType[_, _, _, _]):
    pass


# Partially-bound with omitted parameters
def my_fn4(mt: MyType["Hi there!", ...]):
    pass


# Unbound with omitted parameters
def my_fn5(mt: MyType):
    pass


@fieldwise_init
struct MyComplicatedType[a: Int = 7, /, b: Int = 8, *, c: Int, d: Int = 9]:
    pass


def my_func1(t: MyComplicatedType):
    pass


def my_func2(t: MyComplicatedType[1, ...]):
    pass


def my_func3(t: MyComplicatedType[1, 8, c=_, d=9]):
    pass


@fieldwise_init
struct KeyWordStruct[pos_or_kw: Int, *, kw_only: Int = 10]:
    pass


# Unbind both pos_or_kw and kw_only parameters
def use_kw_struct(k: KeyWordStruct[...]):
    pass


def main():
    # start-has-default-usage
    var instance1 = UseDefault()  # instance of HasDefault[10, 0]
    # end-has-default-usage
    _ = instance1

    # start-partially-bound-example
    comptime StringKeyDict = Dict[String, _]
    var b: StringKeyDict[UInt8] = {"answer": 42}
    # end-partially-bound-example
    _ = b^

    use_kw_struct(KeyWordStruct[10, kw_only=11]())
