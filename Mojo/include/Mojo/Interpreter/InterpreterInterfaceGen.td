//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef INTERPRETERINTERFACEGEN_TD
#define INTERPRETERINTERFACEGEN_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// BytecodeInterpreterOpInterface
//===----------------------------------------------------------------------===//

def BytecodeInterpreterOpInterface :
    OpInterface<"BytecodeInterpreterOpInterface"> {
  let description = [{
    This interface defines an API for operations to generate custom bytecode and
    leverage it to be interpreted at compile time. Operation specify bytecode
    size requirements, define a hook for generating bytecode, and a hook for
    being interpreted.
  }];

  let cppNamespace = "::M";

  let methods = [
    StaticInterfaceMethod<[{
      Get a function pointer that interprets the operation given its constant
      inputs. An operation specific payload is provided to the hook.
    }], "OpBytecodeGenerator", "getBytecodeGenerator">
  ];
}

//===----------------------------------------------------------------------===//
// MemoryableTypeInterface
//===----------------------------------------------------------------------===//

def MemoryableTypeInterface : TypeInterface<"MemoryableTypeInterface"> {
  let description = [{
    This interface allows types to provide an API with which to read and write
    the constant values of its type as raw data. This allows types to plug into
    the interpreter's memory model and enables, for example, writing
    `struct<(i32, i32)>` to a piece of memory, then bitcasting that to read as an
    `i64`.
  }];

  let cppNamespace = "::M";

  let methods = [
    InterfaceMethod<[{
      Write the raw contents of the attribute to the provided chunk of memory.
    }], "ErrorOrSuccess", "writeTo", (ins "TypedAttr":$value,  "int64_t":$addr,
                                          "InterpreterState &":$state)>,
    InterfaceMethod<[{
      Materialize an instance of this type by reading the raw contents of an
      attribute that represents a value of this type from the provided chunk of
      memory.
    }], "ErrorOr<TypedAttr>", "readFrom", (ins "int64_t":$addr,
                                               "InterpreterState &":$state)>
  ];
}

#endif // INTERPRETERINTERFACEGEN_TD
