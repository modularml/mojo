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
// Runtime representation of TargetInfoAttr, DeviceRefAttr, DeviceSpecAttr and
// DeviceCollectionAttr. These can be used at runtime to both confirm the
// runtime environment matches that expected at compile time, and help
// models establish which actual runtime devices correspond to each
// 'abstract' device they assumed at compile time.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEVICE_SPECS_H
#define SUPPORT_DEVICE_SPECS_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/JSON.h"
#include "llvm/TargetParser/Triple.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace M {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Result of decoding a signed LLVM feature string.
struct DecodedFeatures {
  std::vector<std::string> enabled;  ///< Plain names of enabled features.
  std::vector<std::string> disabled; ///< Plain names of disabled features.
};

/// Decodes a signed LLVM feature string (e.g. "+avx2,+bmi1,-avx512f") into
/// separate enabled and disabled plain-name lists. Applies last-wins semantics
/// so duplicate names resolve to the final sign. Unsigned names (no prefix)
/// are treated as enabled for backward compat with older serialized data.
/// Returns an error if a sign prefix has no name after it.
ErrorOr<DecodedFeatures> decodeFeatures(StringRef encodedFeatures);

//===----------------------------------------------------------------------===//
// TargetInfo
//===----------------------------------------------------------------------===//

/// Runtime analogue of part of TargetInfoAttr.
///
/// See also HostMachineInfo and PackageTarget which have similar
/// structure.
struct TargetInfo {
  llvm::Triple triple;
  std::string arch;

  /// Enabled CPU features, as plain unsigned names (e.g. "avx2").
  std::vector<std::string> features;

  /// CPU features that are explicitly disabled (e.g. "avx512f"). These are
  /// present in the CPU model's defaults but unavailable at runtime.
  /// See getHostTargetInfo() for why this field exists.
  std::vector<std::string> disabledFeatures;

  /// The target ABI name (e.g. RISC-V's `lp64d`), mirroring `TargetInfoAttr`'s
  /// `abi`. Empty means no ABI constraint.
  std::string abi;

  TargetInfo(llvm::Triple triple = llvm::Triple(""), std::string arch = {},
             std::vector<std::string> &&features = {},
             std::vector<std::string> &&disabledFeatures = {},
             std::string abi = {})
      : triple(std::move(triple)), arch(std::move(arch)),
        features(std::move(features)),
        disabledFeatures(std::move(disabledFeatures)), abi(std::move(abi)) {}

  /// Serializes this target info to JSON.
  void serializeToJSON(llvm::json::OStream &json) const;
  std::string serializeToJSON() const;

  /// Returns the target info deserialized from JSON.
  static ErrorOr<TargetInfo> deserializeFromJSON(const llvm::json::Value *json);
  static ErrorOr<TargetInfo> deserializeFromJSON(StringRef json);

  /// Returns error if this target info does not satisfy the assumptions
  /// in required.
  ErrorOrSuccess checkSatisfiesRequirements(const TargetInfo &required) const;
};

/// Encodes a TargetInfo's features into the signed string form expected by
/// LLVM (e.g. "+avx2,+bmi1,-avx512f").
std::string encodeFeatures(const TargetInfo &targetInfo);

//===----------------------------------------------------------------------===//
// DeviceRef
//===----------------------------------------------------------------------===//

/// Compile time index of a particular abstract device amongst others with the
/// same or similar target info. The relationship between the device id
/// used in an #M.device_spec and an actual physical device (eg a CUDA device
/// id) is determined at runtime.
using DeviceId = uint64_t;

struct DeviceRef {
  std::string label;
  DeviceId id;

  explicit DeviceRef(std::string label = {}, DeviceId id = 0)
      : label(std::move(label)), id(id) {}

  /// Serialized this device ref to JSON.
  void serializeToJSON(llvm::json::OStream &json) const;

  /// Returns the device ref deserialized from JSON.
  static ErrorOr<DeviceRef> deserializeFromJSON(const llvm::json::Value *json);

  /// Returns device reference in compact string form.
  std::string toString() const;

  bool operator==(const DeviceRef &that) const {
    return std::tie(label, id) == std::tie(that.label, that.id);
  }
};

/// Well known device labels. Closed set -- adding a new label requires
/// updating `M::Engine::Context::create` (max/internal/lib/API/c/
/// EngineContext.cpp) to accept it, otherwise the C API will reject
/// any device that carries the new value.
constexpr const char *kCPULabel = "cpu";
constexpr const char *kGPULabel = "gpu";
constexpr const char *kNPULabel = "npu";

} // namespace M

namespace llvm {

template <>
struct DenseMapInfo<M::DeviceRef> {
  static inline M::DeviceRef getEmptyKey() { return M::DeviceRef(); }
  static inline M::DeviceRef getTombstoneKey() { return M::DeviceRef("", -1); }
  static unsigned getHashValue(const M::DeviceRef &ref) {
    return hash_value(std::make_pair(ref.label, ref.id));
  }
  static bool isEqual(const M::DeviceRef &lhs, const M::DeviceRef &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

namespace M {

//===----------------------------------------------------------------------===//
// DeviceSpec
//===----------------------------------------------------------------------===//

/// Runtime analogue of DeviceSpecAttr.
struct DeviceSpec {
  DeviceRef ref;
  TargetInfo target;

  /// Serialized this device spec to JSON.
  void serializeToJSON(llvm::json::OStream &json) const;
  std::string serializeToJSON() const;

  /// Returns the device spec deserialized from JSON.
  static ErrorOr<DeviceSpec> deserializeFromJSON(const llvm::json::Value *json);
  static ErrorOr<DeviceSpec> deserializeFromJSON(StringRef json);
};

/// A shared, immutable `DeviceSpec`. A spec owns a `TargetInfo` -- a triple,
/// an arch string and two feature vectors -- so copying one per tensor is far
/// from free, and on the host path `getHostTargetInfo()` fills those vectors
/// in. The runtime keeps one spec per configured device for the session and
/// hands out this handle instead; consumers that only compare `ref` pay a
/// refcount rather than a dozen allocations.
using DeviceSpecRef = std::shared_ptr<const DeviceSpec>;

//===----------------------------------------------------------------------===//
// DeviceSpecCollection
//===----------------------------------------------------------------------===//

/// Runtime analogue of DeviceSpecCollectionAttr.
struct DeviceSpecCollection {
  DeviceRef host;
  std::vector<DeviceSpec> devices;

  /// Serializes this devices spec collection to JSON.
  void serializeToJSON(llvm::json::OStream &json) const;
  std::string serializeToJSON() const;

  /// Returns the device spec collection deserialized from JSON.
  static ErrorOr<DeviceSpecCollection>
  deserializeFromJSON(const llvm::json::Value *json);
  static ErrorOr<DeviceSpecCollection> deserializeFromJSON(StringRef json);

  /// Returns the device spec which matches device reference, or an error if
  /// no match.
  ErrorOr<const DeviceSpec *> findDeviceSpec(const DeviceRef &ref) const;

  /// Returns the device spec corresponding to the 'host' in this collection.
  /// (Not to be confused with the actual host machine.)
  const DeviceSpec &getHostDeviceSpec() const;
};

//===----------------------------------------------------------------------===//
// SIMD Width
//===----------------------------------------------------------------------===//

/// Returns true if `feature` is enabled in a comma-separated signed LLVM
/// feature string (e.g. "+avx2,+bmi1,-avx512f"). Scans tokens exactly;
/// disabled tokens ("-feature") and prefix matches (e.g. "avx512" against
/// "+avx512f") both return false.
bool hasFeature(StringRef llvmFeatureStr, StringRef feature);

/// Returns the SIMD bit width implied by a single plain (unsigned) feature
/// name, e.g. "avx512f" → 512, "avx2" → 256, "hvx-length128b" → 1024,
/// anything else → 128. The result is in bits, whatever unit the feature
/// spells its length in.
///
/// This feeds `TargetInfoAttr`'s `simd_bit_width`, which is the width of one
/// vector register as seen by a *single execution context* — one CPU thread,
/// one GPU lane, one DSP hardware thread. It is not the width of the whole
/// vector unit and it is not scaled by how many contexts run together, which
/// is why every GPU target declares 128 (the widest per-lane vector access)
/// rather than a warp-sized number, and why a DSP whose one thread owns a
/// 128-byte vector register declares 1024. A target's entry answers "how much
/// work fits in one register here", so that a kernel deriving its vector shape
/// from the field lands on a shape the target can issue.
size_t simdWidthFromFeature(StringRef plainFeature);

/// Returns the SIMD bit width implied by a comma-separated signed LLVM feature
/// string (e.g. "+avx2,+bmi1,-avx512f"). Disabled features (leading '-') are
/// ignored. Assumes no duplicate feature names; use decodeFeatures first if
/// the source string may contain duplicates.
size_t simdWidthFromFeatures(StringRef llvmFeatureStr);

/// Returns the SIMD bit width implied by a list of plain (unsigned) feature
/// names.
size_t simdWidthFromFeatures(ArrayRef<std::string> plainFeatures);

} // namespace M

#endif // SUPPORT_DEVICE_SPECS_H
