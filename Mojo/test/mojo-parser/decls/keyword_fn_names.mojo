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

# RUN: %parse-mojo-isolated -verify-diagnostics %s

# expected-error @+1 {{'_' cannot be used as a function name in this context}}
def _():
    pass

# expected-error @+1 {{'and' cannot be used as a function name in this context}}
def and():
    pass

# expected-error @+1 {{'as' cannot be used as a function name in this context}}
def as():
    pass

# expected-error @+1 {{'assert' cannot be used as a function name in this context}}
def assert():
    pass

# expected-error @+1 {{'async' cannot be used as a function name in this context}}
def async():
    pass

# expected-error @+1 {{'await' cannot be used as a function name in this context}}
def await():
    pass

# expected-error @+1 {{'break' cannot be used as a function name in this context}}
def break():
    pass

# expected-error @+1 {{'case' cannot be used as a function name in this context}}
def case():
    pass

# expected-error @+1 {{'class' cannot be used as a function name in this context}}
def class():
    pass

# expected-error @+1 {{'comptime' cannot be used as a function name in this context}}
def comptime():
    pass

# expected-error @+1 {{'continue' cannot be used as a function name in this context}}
def continue():
    pass

# expected-error @+1 {{'def' cannot be used as a function name in this context}}
def def():
    pass

# expected-error @+1 {{'del' cannot be used as a function name in this context}}
def del():
    pass

# expected-error @+1 {{'elif' cannot be used as a function name in this context}}
def elif():
    pass

# expected-error @+1 {{'else' cannot be used as a function name in this context}}
def else():
    pass

# expected-error @+1 {{'except' cannot be used as a function name in this context}}
def except():
    pass

# expected-error @+1 {{'False' cannot be used as a function name in this context}}
def False():
    pass

# expected-error @+1 {{'finally' cannot be used as a function name in this context}}
def finally():
    pass

# expected-error @+1 {{'for' cannot be used as a function name in this context}}
def for():
    pass

# expected-error @+1 {{'from' cannot be used as a function name in this context}}
def from():
    pass

# expected-error @+1 {{'global' cannot be used as a function name in this context}}
def global():
    pass

# expected-error @+1 {{'if' cannot be used as a function name in this context}}
def if():
    pass

# expected-error @+1 {{'import' cannot be used as a function name in this context}}
def import():
    pass

# expected-error @+1 {{'in' cannot be used as a function name in this context}}
def in():
    pass

# expected-error @+1 {{'is' cannot be used as a function name in this context}}
def is():
    pass

# expected-error @+1 {{'lambda' cannot be used as a function name in this context}}
def lambda():
    pass

# expected-error @+1 {{'match' cannot be used as a function name in this context}}
def match():
    pass

# expected-error @+1 {{'None' cannot be used as a function name in this context}}
def None():
    pass

# expected-error @+1 {{'nonlocal' cannot be used as a function name in this context}}
def nonlocal():
    pass

# expected-error @+1 {{'not' cannot be used as a function name in this context}}
def not():
    pass

# expected-error @+1 {{'or' cannot be used as a function name in this context}}
def or():
    pass

# expected-error @+1 {{'pass' cannot be used as a function name in this context}}
def pass():
    pass

# expected-error @+1 {{'raise' cannot be used as a function name in this context}}
def raise():
    pass

# expected-error @+1 {{'ref' cannot be used as a function name in this context}}
def ref():
    pass

# expected-error @+1 {{'return' cannot be used as a function name in this context}}
def return():
    pass

# expected-error @+1 {{'Self' cannot be used as a function name in this context}}
def Self():
    pass

# expected-error @+1 {{'struct' cannot be used as a function name in this context}}
def struct():
    pass

# expected-error @+1 {{'trait' cannot be used as a function name in this context}}
def trait():
    pass

# expected-error @+1 {{'__extension' cannot be used as a function name in this context}}
def __extension():
    pass

# expected-error @+1 {{'True' cannot be used as a function name in this context}}
def True():
    pass

# expected-error @+1 {{'try' cannot be used as a function name in this context}}
def try():
    pass

# expected-error @+1 {{'var' cannot be used as a function name in this context}}
def var():
    pass

# expected-error @+1 {{'while' cannot be used as a function name in this context}}
def while():
    pass

# expected-error @+1 {{'with' cannot be used as a function name in this context}}
def with():
    pass

# expected-error @+1 {{'yield' cannot be used as a function name in this context}}
def yield():
    pass

# expected-error @+1 {{'__mlir_region' cannot be used as a function name in this context}}
def __mlir_region():
    pass

# expected-error @+1 {{'__get_address_as_uninit_lvalue' cannot be used as a function name in this context}}
def __get_address_as_uninit_lvalue():
    pass

# expected-error @+1 {{'__get_mvalue_as_litref' cannot be used as a function name in this context}}
def __get_mvalue_as_litref():
    pass

# expected-error @+1 {{'__get_litref_as_mvalue' cannot be used as a function name in this context}}
def __get_litref_as_mvalue():
    pass

# expected-error @+1 {{'__get_address_as_owned_value' cannot be used as a function name in this context}}
def __get_address_as_owned_value():
    pass

# expected-error @+1 {{'origin_of' cannot be used as a function name in this context}}
def origin_of():
    pass

# expected-error @+1 {{'type_of' cannot be used as a function name in this context}}
def type_of():
    pass

# expected-error @+1 {{'conforms_to' cannot be used as a function name in this context}}
def conforms_to():
    pass

# expected-error @+1 {{'__functions_in_module' cannot be used as a function name in this context}}
def __functions_in_module():
    pass

# expected-error @+1 {{'__get_current_function_name' cannot be used as a function name in this context}}
def __get_current_function_name():
    pass

# expected-error @+1 {{'__struct_field_ref' cannot be used as a function name in this context}}
def __struct_field_ref():
    pass

# expected-error @+1 {{'__is_run_in_comptime_interpreter' cannot be used as a function name in this context}}
def __is_run_in_comptime_interpreter():
    pass

# expected-error @+1 {{'__match' cannot be used as a function name in this context}}
def __match():
    pass


# Nested free functions are rejected too.
def outer():
    # expected-error @+1 {{'match' cannot be used as a function name in this context}}
    def match():
        pass


# Keyword method names are allowed (reached via `.name()`).
struct Bar:
    def __init__(out self):
        pass

    def match(self):
        pass

    def class(self):
        pass

    def yield(self):
        pass
