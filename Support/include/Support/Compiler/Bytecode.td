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

#ifndef SUPPORT_BYTECODE_TD
#define SUPPORT_BYTECODE_TD

include "mlir/IR/BytecodeBase.td"

def String :
  WithParser<"succeeded($_reader.readString($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeOwnedString($_getter)",
  WithType<"StringRef">>>>;

class ConcreteType<string type> :
  WithParser<"succeeded($_reader.readType($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeType($_getter)",
  WithType<type>>>>;

class ConcreteAttribute<string type> :
  WithParser<"succeeded($_reader.readAttribute($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeAttribute($_getter)",
  WithType<type>>>>;

class ConcreteOptionalAttribute<string type> :
  WithParser<"succeeded($_reader.readOptionalAttribute($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeOptionalAttribute($_getter)",
  WithType<type>>>>;

class OptionalAttribute<string type> :
  WithParser<"succeeded($_reader.readOptionalAttribute($_var))",
  WithPrinter<"$_writer.writeOptionalAttribute($_getter)",
  WithType<type>>>;

class IntegralAttr<string type> :
  WithParser<"succeeded(M::readIntegral<" # type # ">($_reader, $_var))",
  WithPrinter<"M::writeIntegral($_writer, $_getter)",
  WithType<type>>>;

class EnumAttr<string type> : IntegralAttr<type>;

class ConcreteAttrArray<string type> :
  WithParser<"succeeded($_reader.readAttributes($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeAttributes($_getter)",
  WithType<"SmallVector<" # type # ">">>>>;

class ConcreteTypeArray<string type> :
  WithParser<"succeeded($_reader.readTypes($_var))",
  WithBuilder<"$_args",
  WithPrinter<"$_writer.writeTypes($_getter)",
  WithType<"SmallVector<" # type # ">">>>>;

def TypeArray : ConcreteTypeArray<"Type">;

def TypedAttr : ConcreteAttribute<"TypedAttr">;

def StringAttr : ConcreteAttribute<"StringAttr">;

def StringAttrArray : ConcreteAttrArray<"StringAttr">;

def TypedAttrArray : ConcreteAttrArray<"TypedAttr">;

class IntegralAttrArray<string type> :
  WithParser<"succeeded(M::readIntegralArray<" # type # ">($_reader, $_var))",
  WithPrinter<"M::writeIntegralArray($_writer, $_getter)",
  WithType<"SmallVector<" # type # ">">>>;

class EnumAttrArray<string type> : IntegralAttrArray<type>;

class Custom<string type> :
  WithParser<"succeeded(read" # type # "($_reader, $_var))",
  WithPrinter<"write" # type # "($_writer, $_getter)",
  WithType<type>>>;

#endif // SUPPORT_BYTECODE_TD
