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

#include "Mojo/ToolCommon/CLOptions.h"
#include "Mojo/ExecutionEngine/ExecutionEngine.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
using namespace KGEN;

//===--------------------------------------------------------------------===//
// CommandLineFunc implementation
//===--------------------------------------------------------------------===//

/// Check that `t` returns exactly one type, and it's of type
/// `!pop.array<0, i1>`, which is what mojo uses as it's 'None' type.
static bool returnTypeIsMojoNone(FunctionType t) {
  if (t.getNumResults() != 1)
    return false;

  Type res = t.getResult(0);
  auto array = dyn_cast<POP::ArrayType>(res);
  // Not an array.
  if (!array)
    return false;
  auto intTy = dyn_cast<IntegerType>(array.getElementType());
  // Not an array of integers.
  if (!intTy)
    return false;
  // Not an array of i1.
  if (intTy.getIntOrFloatBitWidth() != 1)
    return false;
  // List is not length 0, or length is unresolved.
  if (auto len = array.getResolvedSize(); !len || *len != 0)
    return false;

  // OK, it is the thing we want.
  return true;
}

ErrorOrSuccess
CommandLineFunc::verifyFuncSignature(mlir::FunctionType funcType) const {
  if (signature == "f32()") {
    if (funcType.getNumInputs() != 0 || funcType.getNumResults() != 1 ||
        !isa<Float32Type>(funcType.getResult(0))) {
      return Error("command-line specified signature does not match the IR "
                   "signature, expected " +
                   mlir::debugString(funcType) + ", but got " + signature);
    }
    return M::success();
  } else if (signature == "()") {
    if (!(returnTypeIsMojoNone(funcType) || funcType.getNumResults() == 0) ||
        funcType.getNumInputs() != 0) {
      return Error("command-line specified signature does not match the IR "
                   "signature, expected " +
                   mlir::debugString(funcType) + ", but got " + signature);
    }
    return M::success();
  } else if (signature == "index()") {
    if (funcType.getNumInputs() != 0 || funcType.getNumResults() != 1 ||
        !isa<IndexType>(funcType.getResult(0))) {
      return Error("command-line specified signature does not match the IR "
                   "signature, expected " +
                   mlir::debugString(funcType) + ", but got " + signature);
    }
    return M::success();
  } else if (signature == "f32(f32)") {
    if (funcType.getNumInputs() != 1 || funcType.getNumResults() != 1 ||
        !isa<Float32Type>(funcType.getResult(0)) ||
        !isa<Float32Type>(funcType.getInput(0))) {
      std::string ktype;
      llvm::raw_string_ostream os(ktype);
      os << funcType;
      return Error("command-line specified signature does not match the IR "
                   "signature, expected " +
                   ktype + ", but got " + signature);
    }
    return M::success();
  }

  return Error("unhandled signature: " + signature);
}

ErrorOrSuccess
CommandLineFunc::executeAndPrint(KGEN::CompiledFunc &compiledFunc) const {
  if (signature == "f32()") {
    printf("--- '%s' returned %f\n", name.c_str(),
           compiledFunc.invoke<float>());
    return M::success();
  } else if (signature == "index()") {
    printf("--- '%s' returned %ld\n", name.c_str(),
           compiledFunc.invoke<ssize_t>());
    return M::success();
  } else if (signature == "()") {
    compiledFunc.invoke<void>();
    printf("--- '%s' finished\n", name.c_str());
    return M::success();
  } else if (signature == "f32(f32)") {
    // TODO: We could parse a float for this, but for now just pass in 1.0 for
    //       all floats.
    printf("--- '%s' returned %f\n", name.c_str(),
           compiledFunc.invoke<float, float>(1.0));
    return M::success();
  }

  return Error("unhandled signature: " + signature);
}

bool CommandLineFuncParser::parse(llvm::cl::Option &o, StringRef argName,
                                  StringRef argValue, CommandLineFunc &val) {
  // Match a function name and signature, of the form: `name:signature`. This
  // check also ensures that "name" supports '::' tokens, which may be used for
  // scope signifiers.
  static llvm::Regex funcAndSignatureMatcher("(.*[^:]):([^:].*)");

  // Check if the value contains the name and signature.
  SmallVector<StringRef> matches;
  if (funcAndSignatureMatcher.match(argValue, &matches)) {
    val.name = matches[1];
    val.signature = matches.back();
    return false;
  }

  // Otherwise, if we don't have a signature, the value is the name.
  val.name = argValue;
  return false;
}

bool KGENCLOptionsParser::parse(llvm::cl::Option &o, StringRef argName,
                                StringRef argValue, Command &val) {
  if (argName == "elaborate") {
    if (argValue == "no-use-parametric-interpreter")
      val = Command::kElaborateNoUseParametricInterpreter;
    else if (argValue == "use-parametric-interpreter")
      val = Command::kElaborateUseParametricInterpreter;
    else
      val = Command::kElaborate;

    return false;
  }
  if (argName == "emit") {
    if (argValue == "llvm")
      val = Command::kEmitLLVM;
    else if (argValue == "llvm-bitcode")
      val = Command::kEmitLLVMBitcode;
    else if (argValue == "llvm-opt")
      val = Command::kEmitLLVMOpt;
    else if (argValue == "llvm-opt-bitcode")
      val = Command::kEmitLLVMOptBitcode;
    else if (argValue == "asm")
      val = Command::kEmitAssembly;
    else if (argValue == "asm-verbose")
      val = Command::kEmitAssemblyVerbose;
    else if (argValue == "object")
      val = Command::kEmit;
    else if (argValue == "shared-lib")
      val = Command::kEmitSharedObject;
    else if (argValue == "header")
      val = Command::kEmitHeader;
    else if (argValue.empty())
      val = Command::kEmit;
    else
      return o.error("unsupported 'emit' option value '" + argValue + "'");

    return false;
  }

  if (argName == "execute") {
    val = Command::kExecute;
    return false;
  }

  if (argName == "lsp") {
    if (argValue == "no-dump")
      val = Command::kLSPNoDump;
    else if (argValue.empty())
      val = Command::kLSP;
    else
      return o.error("unsupported 'lsp' option value '" + argValue + "'");

    return false;
  }
  return o.error("unsupported option '" + argName + "'");
}

llvm::ManagedStatic<KGENPassCLOptions::PassOptions>
    KGENPassCLOptions::passOptions;
