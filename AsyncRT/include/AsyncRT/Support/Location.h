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

#ifndef ASYNCRT_SUPPORT_LOCATION_H
#define ASYNCRT_SUPPORT_LOCATION_H

#include "Support/RCRef.h"

#include <cstdint>
#include <string>

namespace M {
class Error;
}

namespace M::AsyncRT {
class EncodedLocation;
class EncodedDiagnostic;

/// This represents a "decoded" location that is usable for diagnostic emission
/// and other processing.  This object is relatively heavy-weight that is
/// created on demand when reporting an error.  Creation of an error is lighter
/// weight, typically using EncodedLocation.
///
class DecodedLocation {
public:
  std::string filename;
  int line = -1;
  int column = -1;
};

/// This virtual base class is implemented by things that produce
/// `EncodedLocation`s, showing how to decode them.
class LocationDecoder {
public:
  virtual DecodedLocation decode(const EncodedLocation &loc) const = 0;

  /// Add a new reference to this object.
  virtual void addRef() const = 0;

  /// Add a new reference to this object.
  virtual void dropRef() const = 0;

  virtual ~LocationDecoder() = default;

private:
  virtual void VtableAnchor();
};

/// This class is an opaque location token that is efficiently constructible,
/// but needs conversion into a Location before it can be used for reporting.
class EncodedLocation {
public:
  EncodedLocation(intptr_t data, RCRef<LocationDecoder> decoder)
      : data(data), decoder(std::move(decoder)) {}
  EncodedLocation(EncodedLocation &&other) = default;
  EncodedLocation &operator=(EncodedLocation &&other) = default;

  /// Decode the location information in this object into a DecodedLocation.
  DecodedLocation decode() const;

  /// Return a copy of this EncodedLocation.
  EncodedLocation copy() const { return EncodedLocation(data, decoder.copy()); }

  intptr_t getData() const { return data; }

private:
  /// Opaque implementation details of this location, only interpretable by the
  /// location handler.
  intptr_t data;

  /// This is an implementation class that can turn the intptr_t token into a
  /// decoded `Location` object.
  RCRef<LocationDecoder> decoder;
};
} // namespace M::AsyncRT

#endif // ASYNCRT_SUPPORT_LOCATION_H
