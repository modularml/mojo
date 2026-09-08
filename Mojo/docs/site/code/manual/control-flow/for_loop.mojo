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
from std.python import Python


def main() raises:
    var states: List = [String("California"), "Hawaii", "Oregon"]
    for state in states:
        print(state)

    var numbers = {42, 0}
    for number in numbers:
        print(number)

    var capitals: Dict[String, String] = {
        "California": "Sacramento",
        "Hawaii": "Honolulu",
        "Oregon": "Salem",
    }

    for var state in capitals:
        print(t"{capitals[state]}, {state}")

    for item in capitals.items():
        print(t"{item.value}, {item.key}")

    for i in range(5):
        print(i, end=", ")

    # Extra print statements added to separate the output of the different loops
    print()

    for i in range(5):
        if i == 3:
            continue
        print(i, end=", ")

    print()

    for i in range(5):
        if i == 3:
            break
        print(i, end=", ")

    print()

    for i in range(5):
        print(i, end=", ")
    else:
        print("\nFinished executing 'for' loop")

    print()

    var empty = List[Int]()
    for i in empty:
        print(i)
    else:
        print("Finished executing 'for' loop")

    var animals: List = ["cat", "aardvark", "hippopotamus", "dog"]
    for animal in animals:
        if animal == "dog":
            print("Found a dog")
            break
    else:
        print("No dog found")

    # Using references to modify a list item in a loop
    var values: List = [1, 4, 7, 3, 6, 11]
    for ref value in values:
        if value % 2 != 0:
            value -= 1
    print(values)

    # from python import Python

    # Iterate over a mixed-type Python dictionary
    var py_dict = Python.evaluate("{'a': 1, 'b': 2.71828, 'c': 'sushi'}")
    for py_key in py_dict:  # Each key is of type "PythonObject"
        print(py_key, py_dict[py_key])

    # Iterate over a mixed-type Python dictionary using items()
    py_dict = Python.evaluate("{'a': 1, 'b': 2.71828, 'c': 'sushi'}")
    for py_tuple in py_dict.items():  # Each 2-tuple is of type "PythonObject"
        print(py_tuple[0], py_tuple[1])
