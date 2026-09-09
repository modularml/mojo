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
//
// This file defines an interface that mojo-tblgen backends can use to register
// themselves with the driver executable.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TOOLS_DRIVERTBLGEN_BACKENDREGISTRY_H
#define SUPPORT_TOOLS_DRIVERTBLGEN_BACKENDREGISTRY_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <functional>
#include <string>
#include <vector>

namespace llvm {
class raw_ostream;
class RecordKeeper;
} // namespace llvm

namespace M {

/// Each mojo-tblgen backend registers a callback function with this signature.
/// If the driver is invoked with their backend name, it calls the registered
/// function, passing along the output stream and the parsed TableGen records.
using BackendFunction =
    std::function<bool(llvm::raw_ostream &, const llvm::RecordKeeper &)>;

/// A simple wrapper for each backend.
struct Backend {
  /// The name of the option used to invoke this backend. For example, a name
  /// "do-the-thing" results in users being able to invoke
  /// `mojo-tblgen -do-the-thing`.
  std::string name;
  /// A description of this backend. This is output as help text when invoking
  /// `mojo-tblgen --help`.
  std::string description;
  /// The actual implementation of the backend's logic. This function is called
  /// when the backend is selected by the driver.
  const BackendFunction function;
};

/// A collection of registered backends.
class BackendRegistry {
public:
  /// Register the given callback function under the given name and description.
  void addBackend(llvm::StringRef name, llvm::StringRef description,
                  const BackendFunction &function) {
    backends.push_back({name.str(), description.str(), function});
  }

  /// Get all registered backends.
  llvm::ArrayRef<Backend> getBackends() const { return backends; }

private:
  std::vector<Backend> backends;
};
} // namespace M

#endif // SUPPORT_TOOLS_DRIVERTBLGEN_BACKENDREGISTRY_H
