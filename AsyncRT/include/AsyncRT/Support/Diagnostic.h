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
// Diagnostics are combinations of an error message + location information.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_SUPPORT_DIAGNOSTIC_H
#define ASYNCRT_SUPPORT_DIAGNOSTIC_H

#include "AsyncRT/Support/Location.h"
#include "Support/Error.h"
#include <variant>

namespace M::AsyncRT {

/// This is a combination of an `Error` message with an encoded location.  It is
/// relatively efficient to pass around, but its location must be decoded before
/// it can be interpreted.
class EncodedDiagnostic {
public:
  EncodedDiagnostic(Error message, EncodedLocation location)
      : message(std::move(message)), location(std::move(location)) {}
  EncodedDiagnostic(EncodedDiagnostic &&) = default;

  /// Access the message in the diagnostic.
  const Error &getMessage() const { return message; }
  Error &getMessage() { return message; }

  /// Access the location in the diagnostic.
  const EncodedLocation &getLocation() const { return location; }
  EncodedLocation &getLocation() { return location; }

  /// Decode the compressed location into a `DecodedLocation` for rendering.
  DecodedLocation decodeLocation() const { return location.decode(); }

private:
  Error message;
  EncodedLocation location;
};

/// Container holding either an EncodedDiagnostic or a Value. Provides locations
/// for errors compared with ErrorOr.
template <typename Value>
class [[nodiscard]] ErrorDiagnosticOr {
public:
  bool isError() const {
    return std::holds_alternative<EncodedDiagnostic>(data);
  }

  EncodedDiagnostic takeError() { return std::move(std::get<0>(data)); }
  Value takeValue() { return std::move(std::get<1>(data)); }

  ErrorDiagnosticOr(EncodedDiagnostic err) : data(std::move(err)) {}
  ErrorDiagnosticOr(Value val) : data(std::move(val)) {}

private:
  std::variant<EncodedDiagnostic, Value> data;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_SUPPORT_DIAGNOSTIC_H
