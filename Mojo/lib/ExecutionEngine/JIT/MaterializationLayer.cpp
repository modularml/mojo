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

#include "Mojo/ExecutionEngine/JIT/MaterializationLayer.h"
#include "Support/ErrorOr.h"
#include "llvm/ExecutionEngine/Orc/COFFPlatform.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Debugging/DebugInfoSupport.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Base64.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// MaterializationLayer
//===----------------------------------------------------------------------===//

MaterializationLayer::MaterializationLayer(LayerKind kind,
                                           llvm::orc::ExecutionSession &sess,
                                           const llvm::DataLayout &dl,
                                           AddToSearchOrderFn add)
    : session(sess), dataLayout(dl), addToSearchOrder(std::move(add)),
      kind(kind) {}

ErrorOr<llvm::orc::JITDylib *>
MaterializationLayer::getOrCreateDylib(StringRef libName) {
  if (llvm::orc::JITDylib *dylib = session.getJITDylibByName(libName))
    return dylib;

  auto dylibOr = session.createJITDylib(libName.str());
  if (!dylibOr)
    return toModularError(dylibOr.takeError());
  llvm::orc::JITDylib &dylib = *dylibOr;

  // Add the dylib to the search order.
  if (auto err = addToSearchOrder(libName, &dylib))
    return err.takeError();

  return &dylib;
}

llvm::orc::SymbolStringPtr
MaterializationLayer::mangleAndIntern(StringRef name) {
  std::string mangledName;
  llvm::raw_string_ostream mangledNameStream(mangledName);
  llvm::Mangler::getNameWithPrefix(mangledNameStream, name, dataLayout);
  return session.intern(mangledName);
}
