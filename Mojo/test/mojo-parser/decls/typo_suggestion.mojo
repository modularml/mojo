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

# Near-miss suggestions for unknown declarations via edit distance.


def use(x: Int):
    pass


def local_typo():
    var count = 1
    # expected-error @+1 {{use of unknown declaration 'coun'; did you mean 'count'?}}
    use(coun)


def param_typo(value: Int):
    # expected-error @+1 {{use of unknown declaration 'valu'; did you mean 'value'?}}
    use(valu)


def outer_scope_typo():
    var module_level = 7

    def inner():
        # expected-error @+1 {{use of unknown declaration 'module_leve'; did you mean 'module_level'?}}
        use(module_leve)

    inner()


# No suggestion when nothing in scope is close enough.
def no_near_miss():
    # expected-error @+1 {{use of unknown declaration 'zzzzz'}}
    use(zzzzz)


# Short names are not suggested against each other.
def short_name(x: Int):
    # expected-error @+1 {{use of unknown declaration 'y'}}
    use(y)


# Tied nearest neighbors suppress the suggestion rather than guessing.
def ambiguous_tie():
    var fooa = 1
    var foob = 2
    # expected-error @+1 {{use of unknown declaration 'foox'}}
    use(foox)
    use(fooa)
    use(foob)


# Hardcoded rename still wins over a coincidental near miss.
def type_of_rename(type_of_x: Int):
    # expected-error @+1 {{use of unknown declaration 'typeof'; did you mean 'type_of'?}}
    _ = typeof
