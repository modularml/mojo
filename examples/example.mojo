# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
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
//This example demonstrates a simple Mojo class with a property, an initializer, and a method. It prints the value of the property to the console.
// Define a class named Person.
class Person {
  // Declare a property named name of type String.
  let name: String

  // Declare a property named age of type Int.
  let age: Int

  // Define a constructor that takes two arguments.
  init(name: String, age: Int) {
    self.name = name
    self.age = age
  }

  // Define a method named sayHello that takes a parameter.
  func sayHello(name: String) {
    Console.log("Hello, " + name + "!")
  }

  // Define a method named getName that returns the value of the name property.
  func getName() -> String {
    return name
  }

  // Define a method named setAge that takes an integer parameter and updates the age property.
  func setAge(age: Int) {
    self.age = age
  }
}

// Create an instance of Person with initial values.
let person = Person(name: "John", age: 30)

// Call the sayHello method with a parameter.
person.sayHello(name: "Jane")

// Call the getName method to retrieve the value of the name property.
let name = person.getName()
Console.log("The person's name is: " + name)

// Call the setAge method to update the age property.
person.setAge(age: 31)

// Call the sayHello method again to demonstrate that the age property has been updated.
person.sayHello(name: "Jane")
