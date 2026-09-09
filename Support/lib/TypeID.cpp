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

#include "Support/TypeID.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstddef>
#include <mutex>
#include <string_view>

#define DEBUG_TYPE "typeids"

using namespace M;

Detail::RawTypeID Detail::TypeInfoTable::getSlow(std::string_view typeName,
                                                 ValueDestructorFn destructor) {
  std::lock_guard<std::mutex> l(mu);
  auto itr = ids.find(typeName);
  if (itr != ids.end())
    return itr->second;

  size_t id = entries.emplace_back(typeName, destructor);
  assert(id != Detail::kInvalidRawTypeID && "too many type ids registered");
  LDBG() << "Registering type " << typeName << " with " << id;
  [[maybe_unused]] auto pair = ids.try_emplace(typeName, id);
  assert(pair.second && "already registered type");
  return id;
}

Detail::RawTypeID TypeID::getSlow(std::string_view typeName,
                                  ValueDestructorFn destructorFn) {
  return Detail::TypeInfoTable::getSingleton().getSlow(typeName, destructorFn);
}

#ifdef MODULAR_DEBUG
void TypeID::printErrorIfNotEqual(TypeID expected, StringRef context) const {
  if (id == expected.id)
    return;
  llvm::errs() << context << ": object has actual runtime type '"
               << getTypeName()
               << "' however it was expected at compile time to have type '"
               << expected.getTypeName() << "'\n";
}
#endif
