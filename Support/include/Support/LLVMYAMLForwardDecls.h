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
// This file forward defines macros for forward-declaring YAML traits normally
// found in llvm/Support/YAMLTraits.h.  YAMLTraits.h is a large header and has
// a significant impact on compile time -- while including it is unavoidable in
// implementation files, most header files only need a tiny amount of what it
// declares.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_LLVM_YAML_FORWARD_DECLS_H
#define SUPPORT_LLVM_YAML_FORWARD_DECLS_H

namespace llvm {
class StringRef;
class raw_ostream;
namespace yaml { // NOLINT, upstream namespace
class IO;
template <class T>
struct MappingTraits;
template <typename T, typename Enable>
struct ScalarTraits;
template <typename T, typename Enable>
struct ScalarEnumerationTraits;
template <typename T, typename Enable>
struct SequenceElementTraits;
template <typename T>
struct CustomMappingTraits;
enum class QuotingType;
} // namespace yaml
} // namespace llvm

// This is identical to LLVM_YAML_DECLARE_MAPPING_TRAITS, but we need to pick a
// different name to avoid a macro redefinition warning.
#define LLVM_FWD_YAML_DECLARE_MAPPING_TRAITS(Type)                             \
  namespace llvm {                                                             \
  namespace yaml {                                                             \
  template <>                                                                  \
  struct MappingTraits<Type> {                                                 \
    static void mapping(IO &IO, Type &Obj);                                    \
  };                                                                           \
  }                                                                            \
  }

#define LLVM_FWD_YAML_DECLARE_ENUM_TRAITS(Type)                                \
  namespace llvm {                                                             \
  namespace yaml {                                                             \
  template <>                                                                  \
  struct ScalarEnumerationTraits<Type> {                                       \
    static void enumeration(IO &io, Type &Value);                              \
  };                                                                           \
  }                                                                            \
  }

// N.B.: The non-forward version of this macro takes a MustQuote argument, but
// this one cannot.  Make sure to define mustQuote in your implementation file.
#define LLVM_FWD_YAML_DECLARE_SCALAR_TRAITS(Type)                              \
  namespace llvm {                                                             \
  namespace yaml {                                                             \
  template <>                                                                  \
  struct ScalarTraits<Type, void> {                                            \
    static void output(const Type &Value, void *ctx, raw_ostream &Out);        \
    static StringRef input(StringRef Scalar, void *ctxt, Type &Value);         \
    static QuotingType mustQuote(StringRef);                                   \
  };                                                                           \
  }                                                                            \
  }

#define LLVM_FWD_YAML_IS_FLOW_SEQUENCE_VECTOR(Type)                            \
  namespace llvm {                                                             \
  namespace yaml {                                                             \
  template <>                                                                  \
  struct SequenceElementTraits<Type, void> {                                   \
    static const bool flow = true;                                             \
  };                                                                           \
  }                                                                            \
  }

#define LLVM_FWD_YAML_IS_SEQUENCE_VECTOR(Type)                                 \
  namespace llvm {                                                             \
  namespace yaml {                                                             \
  template <>                                                                  \
  struct SequenceElementTraits<Type, void> {                                   \
    static const bool flow = false;                                            \
  };                                                                           \
  }                                                                            \
  }

#endif // SUPPORT_LLVM_YAML_FORWARD_DECLS_H
