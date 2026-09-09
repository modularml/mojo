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


# RUN: kgen-doc %s | FileCheck %s


# CHECK: "structs"
# CHECK: "constraints": "Must implement TraitA.\nMust implement TraitB."
# CHECK: "description": "This is the main description.\n"
# CHECK: "kind": "struct"
# CHECK: "name": "TestStruct"
# CHECK: "parameters"
# CHECK: "description": "First parameter description."
# CHECK: "name": "param1"
# CHECK: "description": "Second parameter description."
# CHECK: "name": "param2"
# CHECK: "summary": "This is a summary."
struct TestStruct[param1: AnyType, param2: AnyType]:
    """This is a summary.

    This is the main description.

    Parameters:
        param1: First parameter description.
        param2: Second parameter description.

    Constraints:
        Must implement TraitA.
        Must implement TraitB.
    """

    pass


# CHECK: "constraints": ""
# CHECK: "description": "This is the main description.\n"
# CHECK: "kind": "struct"
# CHECK: "name": "MissingConstraintsStruct"
# CHECK: "parameters"
# CHECK: "description": "First parameter description."
# CHECK: "name": "param1"
# CHECK: "summary": "This is a summary."
struct MissingConstraintsStruct[param1: AnyType]:
    """This is a summary.

    This is the main description.

    Constraints:

    Parameters:
        param1: First parameter description.
    """

    pass


# CHECK: "description": "Detailed description\nwith multiple lines."
# CHECK: "summary": "Simple summary."
struct SimpleStruct:
    """Simple summary.

    Detailed description
    with multiple lines.
    """

    pass


# CHECK: "constraints": ""
# CHECK: "description": ""
# CHECK: "parameters": []
# CHECK: "summary": "EmptySectionStruct text."
struct EmptySectionStruct:
    """EmptySectionStruct text.

    Parameters:

    Constraints:
    """

    pass


# CHECK: "constraints": ""
# CHECK: "description": ""
# CHECK: "parameters": []
# CHECK: "summary": "BlankSectionStruct text."
struct BlankSectionStruct:
    """BlankSectionStruct text.

    Parameters:

    Constraints:

    """

    pass


# CHECK: "constraints": ""
# CHECK: "description": ""
# CHECK: "fields":
# CHECK: "name": "param1"
# CHECK: "summary": ""
# CHECK: "name": "param2"
# CHECK: "summary": ""
# CHECK: "parameters": []
# CHECK: "summary": "Summary text."
struct MalformedStruct:
    """Summary text.

    Parameters:
    - param1: Missing colon
    param2: Bad indentation
    Constraints:
    Invalid constraint format
    """

    var param1: Int
    var param2: String
