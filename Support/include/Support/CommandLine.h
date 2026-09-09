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
// This file is a convenience wrapper around llvm/Support/CommandLine.h.  In
// addition to including it, this defines an `M::cl` namespace analogous to the
// `llvm::cl` namespace with the important types and functions imported.  This
// avoids having massively llvm::cl::opt sorts of qualifications in Modular
// code.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_COMMANDLINE_H
#define SUPPORT_COMMANDLINE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"

namespace M {

namespace cl { // NOLINT, mirrors upstream llvm::cl

using alias = llvm::cl::alias;

template <class DataType, class StorageClass = bool,
          class ParserClass = llvm::cl::parser<DataType>>
using list = llvm::cl::list<DataType, StorageClass, ParserClass>;

template <class DataType, bool ExternalStorage = false,
          class ParserClass = llvm::cl::parser<DataType>>
using opt = llvm::cl::opt<DataType, ExternalStorage, ParserClass>;

using desc = llvm::cl::desc;
using value_desc = llvm::cl::value_desc;

template <class Ty>
inline llvm::cl::initializer<Ty> init(const Ty &Val) {
  return llvm::cl::init(Val);
}
template <typename... OptsTy>
inline llvm::cl::ValuesClass values(OptsTy... Options) {
  return llvm::cl::values(Options...);
}

/// The following set of classes are intended to be used for command line
/// parsing as they provide a destructor to clean up command line options
/// maintained by llvm. Using a llvm::cl::opt will store the option in a global
/// structure. Furthermore, using a llvm::cl::opt with external storage
/// decouples the parser (llvm::cl::opt) and the options themselves.
/// Ideally, developers should write a normal options structure (or class) and
/// then write a separate parser class which ties the command line options to
/// the runtime options.

/// Given the following options structure:

/// struct FooOptions {
/// public:
///   llvm::cl::opt<int> myConfigValue{
///       "config-value", cl::desc("Set your config value"),
///       llvm::cl::init(42)};
/// };

/// We can refactor it as below, where the `FooOptions` structure would stay in
/// the same place, while the `FooCLOptions` can be moved to the
/// tooling/executable file. Also note that the options structure is more
/// verbose in the type and default value compared to the `llvm::cl::opt`
/// representation.

/// struct FooOptions {
/// public:
///   int myConfigValue{42};
/// };

/// struct FooCLOptions {
/// public:
///   FooOptions &options;
///   FooCLOptions(FooOptions &opt) : options(opt) {}
///   M::cl::MOpt<int, true> myConfigValue{"config-value",
///                                        cl::desc("Set your config value"),
///                                        llvm::cl::location(options.configVal)};
/// };

template <class DataType, bool ExternalStorage = false,
          class ParserClass = llvm::cl::parser<DataType>>
class MOpt : public llvm::cl::opt<DataType, ExternalStorage, ParserClass> {

public:
  using llvm::cl::opt<DataType, ExternalStorage, ParserClass>::opt;
  ~MOpt() {
    llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &entries =
        llvm::cl::getRegisteredOptions();
    entries.erase(this->ArgStr);
  }
};

template <class DataType, class StorageClass = bool,
          class ParserClass = llvm::cl::parser<DataType>>
class MListOpt : public llvm::cl::list<DataType, StorageClass, ParserClass> {
public:
  using llvm::cl::list<DataType, StorageClass, ParserClass>::list;
  ~MListOpt() {
    llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &entries =
        llvm::cl::getRegisteredOptions();
    entries.erase(this->ArgStr);
  }
};

template <class DataType, class Storage = bool,
          class ParserClass = llvm::cl::parser<DataType>>
class MBitsOpt : public llvm::cl::bits<DataType, Storage, ParserClass> {
public:
  using llvm::cl::bits<DataType, Storage, ParserClass>::bits;
  ~MBitsOpt() {
    llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &entries =
        llvm::cl::getRegisteredOptions();
    entries.erase(this->ArgStr);
  }
};

} // namespace cl

} // namespace M

#endif
