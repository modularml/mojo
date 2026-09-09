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

#ifndef GENERICML_SUPPORT_DEBUGPRINT_H
#define GENERICML_SUPPORT_DEBUGPRINT_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"

#include <string>
#include <utility>

namespace M {

class TensorSpec;

enum class ResultOutputStyle {
  kCompact, // Display a few elements at the start and end of each dimension
  kFull,    // Display all the elements of result
  kBinary,  // Save results to .bin
  kNone,    // Do not display any tensor details
  kBinaryMaxCheckpoint, // Save results to .max (saves tensor shape and dtype).
};

/// This prints the specified tensor data to stdout.
ErrorOrSuccess printTensor(raw_ostream &os, const void *srcPtr,
                           const TensorSpec &spec, ResultOutputStyle style,
                           unsigned precision = 6);

/// Options, conveyed via a runtime context object, which control the output
/// format for debug printing of tensor values.
struct DebugTensorPrintOptions {
  /// Format to use.
  ResultOutputStyle style = ResultOutputStyle::kCompact;
  /// Precision of textual floating point numbers.
  unsigned precision = 6;
  /// If outputStyle is binary, the directory in which to create the files,
  /// using the 'label' for the basename.
  std::string binaryDir;

  DebugTensorPrintOptions() = default;
  DebugTensorPrintOptions(ResultOutputStyle style, unsigned precision,
                          std::string binaryDir)
      : style(style), precision(precision), binaryDir(std::move(binaryDir)) {}

  /// Prints the tensor with CPU-hosted buffer contents and spec, following
  /// the options of this object, and using label if given to disambiguate
  /// the result.
  ErrorOrSuccess printTensor(const void *buffer, const TensorSpec &spec,
                             StringRef label);
};

} // namespace M

#endif // GENERICML_SUPPORT_DEBUGPRINT_H
