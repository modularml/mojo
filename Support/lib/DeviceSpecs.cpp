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

#include "Support/DeviceSpecs.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#define DEBUG_TYPE "device-specs"

using namespace M;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {
/// An arch or feature name decoded into 'base' and 'version' components.
struct VersionedName {
  // eg "sm_80".
  StringRef fullName;
  // eg "sm".
  StringRef baseName;
  // eg 80.
  uint64_t version;

  VersionedName(StringRef fullName, StringRef baseName, uint64_t version)
      : fullName(fullName), baseName(baseName), version(version) {}

  bool operator<=(const VersionedName &that) const {
    return baseName == that.baseName && version <= that.version;
  }

  static VersionedName decode(StringRef name) {
    // Currently just hard coded and not target specific.
    static StringRef prefix("sm_");
    if (name.find(prefix) == 0) {
      StringRef suffix = name.drop_front(prefix.size());
      uint64_t version;
      if (!suffix.getAsInteger(10, version))
        return VersionedName(name, prefix.drop_back(), version);
    }
    return VersionedName(name, name, 0);
  }
};

} // namespace

bool M::hasFeature(StringRef featureStr, StringRef feature) {
  StringRef remaining = featureStr;
  while (!remaining.empty()) {
    auto [token, rest] = remaining.split(',');
    if (token.size() > 1 && token[0] == '+' && token.drop_front(1) == feature)
      return true;
    remaining = rest;
  }
  return false;
}

std::string M::encodeFeatures(const TargetInfo &ti) {
  std::string result;
  for (StringRef f : ti.features) {
    if (!result.empty())
      result += ',';
    result += '+';
    result += f;
  }
  for (StringRef f : ti.disabledFeatures) {
    if (!result.empty())
      result += ',';
    result += '-';
    result += f;
  }
  return result;
}

ErrorOr<DecodedFeatures> M::decodeFeatures(StringRef encodedFeatures) {
  SmallVector<StringRef> tokens;
  encodedFeatures.split(tokens, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  // This complicated loop does two thingss:
  // 1. Tracks insertion order (do not swap or sort features).
  // 2. Applies last-wins semantics: e.g. "+avx512f,-avx512f" resolves to
  // disabled.

  SmallVector<StringRef> order;
  llvm::StringMap<bool> state; // name → enabled (last wins)
  for (StringRef token : tokens) {
    char sign = token.front();
    StringRef name;
    bool enabled;
    if (sign == '+' || sign == '-') {
      if (token.size() < 2)
        return Error(Twine("ill-formed features: '") + encodedFeatures + "'");
      name = token.drop_front(1);
      enabled = (sign == '+');
    } else {
      // Unsigned names treated as enabled for backward compat.
      name = token;
      enabled = true;
    }
    if (!state.contains(name))
      order.push_back(name);
    state[name] = enabled;
  }

  DecodedFeatures result;
  for (StringRef name : order) {
    if (state[name])
      result.enabled.emplace_back(name);
    else
      result.disabled.emplace_back(name);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// TargetInfo
//===----------------------------------------------------------------------===//

void TargetInfo::serializeToJSON(llvm::json::OStream &json) const {
  json.objectBegin();
  json.attribute("triple", triple.str());
  json.attribute("arch", arch);
  json.attribute("features", features);
  json.attribute("disabledFeatures", disabledFeatures);
  json.attribute("abi", abi);
  json.objectEnd();
}

std::string TargetInfo::serializeToJSON() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  llvm::json::OStream json(os);
  serializeToJSON(json);
  return str;
}

ErrorOr<TargetInfo>
TargetInfo::deserializeFromJSON(const llvm::json::Value *json) {
  const llvm::json::Object *object = json->getAsObject();
  if (!object)
    return Error("ill-formed serialized TargetInfo: expecting object");

  std::optional<StringRef> optTriple = object->getString("triple");
  std::optional<StringRef> optArch = object->getString("arch");
  const llvm::json::Array *array = object->getArray("features");
  if (!optTriple || !optArch || !array)
    return Error("ill-formed serialized TargetInfo: missing attributes");

  TargetInfo result;
  result.triple = llvm::Triple(*optTriple);
  result.arch = *optArch;
  for (const llvm::json::Value &v : *array) {
    std::optional<StringRef> optFeature = v.getAsString();
    if (!optFeature)
      return Error(
          "ill-formed serialized TargetInfo: expecting string feature");
    result.features.emplace_back(*optFeature);
  }
  // Read disabledFeatures if present; absent in older serialized data means
  // empty.
  if (const llvm::json::Array *disabled =
          object->getArray("disabledFeatures")) {
    for (const llvm::json::Value &v : *disabled) {
      std::optional<StringRef> optFeature = v.getAsString();
      if (!optFeature)
        return Error("ill-formed serialized TargetInfo: expecting string "
                     "disabledFeature");
      result.disabledFeatures.emplace_back(*optFeature);
    }
  }
  // Absent in older serialized data means no ABI constraint.
  if (std::optional<StringRef> optABI = object->getString("abi"))
    result.abi = *optABI;
  return result;
}

ErrorOr<TargetInfo> TargetInfo::deserializeFromJSON(StringRef json) {
  llvm::Expected<llvm::json::Value> errOrValue = llvm::json::parse(json);
  if (llvm::Error err = errOrValue.takeError()) {
    return Error(Twine("ill-formed serialized target info: ") +
                 toString(std::move(err)));
  }
  return deserializeFromJSON(&errOrValue.get());
}

/// Returns canonicalized architecture name.
static std::string canonArchName(StringRef archName) {
  // arm64 can also mean aarch64.
  if (archName == "aarch64")
    return "arm64";
  return std::string(archName);
}

/// Returns canonicalized OS type.
static llvm::Triple::OSType canonOSType(llvm::Triple::OSType type) {
  // Darwin and macosx are equivalent.
  if (type == llvm::Triple::OSType::Darwin)
    return llvm::Triple::OSType::MacOSX;
  return type;
}

/// Returns success if provided triple matches required triple, up to
/// heuristics to account for version numbers, vendor names, etc.
static ErrorOrSuccess satisfiesTriple(const llvm::Triple &provided,
                                      const llvm::Triple &required) {
  LDBG() << "provided: " << provided.str() << "\n"
         << "required: " << required.str();

  if (required.str().empty()) {
    // No constraint.
    return success();
  }

  std::string providedArch = canonArchName(provided.getArchName());
  std::string requiredArch = canonArchName(required.getArchName());
  if (required.getArch() != llvm::Triple::ArchType::UnknownArch &&
      providedArch != requiredArch) {
    return Error(Twine("Provided architecture '") + provided.getArchName() +
                 "' does not match required architecture '" +
                 required.getArchName() + "'.");
  }

  llvm::Triple::OSType providedOS = canonOSType(provided.getOS());
  llvm::Triple::OSType requiredOS = canonOSType(required.getOS());
  if (requiredOS != llvm::Triple::OSType::UnknownOS &&
      providedOS != requiredOS) {
    return Error(Twine("Provided OS '") + provided.getOSName() +
                 "' does not match required OS '" + required.getOSName() +
                 "'.");
  }

  return success();
}

/// Returns success if provided arch matches required arch.
static ErrorOrSuccess satisfiesArch(StringRef provided, StringRef required) {
  if (required.empty())
    // No constraint.
    return success();

  auto versionedProvided = VersionedName::decode(provided);
  auto versionedRequired = VersionedName::decode(required);

  if (!(versionedRequired <= versionedProvided)) {
    return Error(Twine("Provided arch '") + provided +
                 "' does not match required arch '" + required + "'.");
  }
  return success();
}

/// Returns success if provided ABI matches required ABI. Unlike arch and
/// features, ABI compatibility is exact-or-nothing: there is no notion of one
/// ABI satisfying a "lesser" one.
static ErrorOrSuccess satisfiesABI(StringRef provided, StringRef required) {
  if (required.empty())
    // No constraint.
    return success();

  if (provided != required) {
    return Error(Twine("Provided ABI '") + provided +
                 "' does not match required ABI '" + required + "'.");
  }
  return success();
}

/// Returns features parsed into map form, where the key is the feature
/// base name.
static llvm::StringMap<VersionedName>
parseFeatures(const std::vector<std::string> &features) {
  llvm::StringMap<VersionedName> map;
  for (const auto &feature : features) {
    // HACK: Ignore features emitted in older versions of system-info when
    // matching against new versions. These are pretty much a given anyways
    // on just about every x64 CPU for the last 20 years and shouldn't be
    // missed.
    // TODO: remove once we clean up old manifests.
    if (feature == "64Bit" || feature == "shstk" || feature == "cmov")
      continue;
    auto versionedFeature = VersionedName::decode(feature);
    map.insert({versionedFeature.baseName, versionedFeature});
  }
  return map;
}

/// Returns success if provided features are a superset of required features,
/// and that no required feature is explicitly disabled.
static ErrorOrSuccess
satisfiesFeatures(const std::vector<std::string> &provided,
                  const std::vector<std::string> &providedDisabled,
                  const std::vector<std::string> &required) {
  if (required.empty())
    return success();

  llvm::StringMap<VersionedName> providedMap = parseFeatures(provided);
  llvm::StringMap<VersionedName> disabledMap = parseFeatures(providedDisabled);
  llvm::StringMap<VersionedName> requiredMap = parseFeatures(required);
  std::string str;
  llvm::raw_string_ostream os(str);
  os << "The following features are required but not provided: ";
  bool anyMissing = false;
  for (const auto &[requiredBase, requiredVersioned] : requiredMap) {
    if (disabledMap.count(requiredBase)) {
      if (anyMissing)
        os << ", ";
      os << requiredVersioned.fullName << " (explicitly disabled)";
      anyMissing = true;
      continue;
    }
    auto providedItr = providedMap.find(requiredBase);
    if (providedItr == providedMap.end()) {
      if (anyMissing)
        os << ", ";
      os << requiredVersioned.fullName;
      anyMissing = true;
    } else if (!(requiredVersioned <= providedItr->second)) {
      if (anyMissing)
        os << ", ";
      os << requiredVersioned.fullName;
      os << " (only have " << providedItr->second.fullName << ")";
      anyMissing = true;
    }
  }
  if (anyMissing)
    return Error(str);
  return success();
}

ErrorOrSuccess
TargetInfo::checkSatisfiesRequirements(const TargetInfo &required) const {
  LDBG_OS([&](raw_ostream &os) {
    os << "provided:\n" << serializeToJSON() << "\n";
    os << "required:\n" << required.serializeToJSON();
  });
  if (auto errOr = satisfiesTriple(triple, required.triple))
    return errOr.takeError();
  if (auto errOr = satisfiesArch(arch, required.arch))
    return errOr.takeError();
  if (auto errOr =
          satisfiesFeatures(features, disabledFeatures, required.features))
    return errOr.takeError();
  if (auto errOr = satisfiesABI(abi, required.abi))
    return errOr.takeError();
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceRef
//===----------------------------------------------------------------------===//

void DeviceRef::serializeToJSON(llvm::json::OStream &json) const {
  json.objectBegin();
  json.attribute("label", label);
  json.attribute("id", id);
  json.objectEnd();
}

ErrorOr<DeviceRef>
DeviceRef::deserializeFromJSON(const llvm::json::Value *json) {
  const llvm::json::Object *object = json->getAsObject();
  if (!object)
    return Error("ill-formed serialized DeviceRef: expecting object");

  std::optional<StringRef> optLabel = object->getString("label");
  std::optional<DeviceId> optId = object->getInteger("id");
  if (!optLabel || !optId)
    return Error("ill-formed serialized DeviceRef: missing attributes");

  DeviceRef result;
  result.label = *optLabel;
  result.id = *optId;
  return result;
}

std::string DeviceRef::toString() const {
  std::string str;
  str += label;
  str += ":";
  str += std::to_string(id);
  return str;
}

//===----------------------------------------------------------------------===//
// DeviceSpec
//===----------------------------------------------------------------------===//

void DeviceSpec::serializeToJSON(llvm::json::OStream &json) const {
  json.objectBegin();
  json.attributeBegin("ref");
  ref.serializeToJSON(json);
  json.attributeEnd();
  json.attributeBegin("target");
  target.serializeToJSON(json);
  json.attributeEnd();
  json.objectEnd();
}

std::string DeviceSpec::serializeToJSON() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  llvm::json::OStream json(os);
  serializeToJSON(json);
  return str;
}

ErrorOr<DeviceSpec> DeviceSpec::deserializeFromJSON(StringRef json) {
  llvm::Expected<llvm::json::Value> errOrValue = llvm::json::parse(json);
  if (llvm::Error err = errOrValue.takeError()) {
    return Error(Twine("ill-formed serialized target info: ") +
                 toString(std::move(err)));
  }
  return deserializeFromJSON(&errOrValue.get());
}

ErrorOr<DeviceSpec>
DeviceSpec::deserializeFromJSON(const llvm::json::Value *json) {
  const llvm::json::Object *object = json->getAsObject();
  if (!object)
    return Error("ill-formed serialized DeviceSpec: expecting object");

  const llvm::json::Value *ref = object->get("ref");
  const llvm::json::Value *target = object->get("target");
  if (!ref || !target)
    return Error("ill-formed serialized DeviceSpec: missing attributes");

  auto refOr = DeviceRef::deserializeFromJSON(ref);
  if (refOr) {
    return Error(Twine("ill-formed serialized DeviceSpec: ") +
                 refOr.getError());
  }

  auto targetOr = TargetInfo::deserializeFromJSON(target);
  if (targetOr) {
    return Error(Twine("ill-formed serialized DeviceSpec: ") +
                 targetOr.getError());
  }

  DeviceSpec result;
  result.ref = std::move(*refOr);
  result.target = std::move(*targetOr);
  return result;
}

//===----------------------------------------------------------------------===//
// DeviceSpecCollection
//===----------------------------------------------------------------------===//

void DeviceSpecCollection::serializeToJSON(llvm::json::OStream &json) const {
  json.objectBegin();
  json.attributeBegin("host");
  host.serializeToJSON(json);
  json.attributeEnd();
  json.attributeBegin("devices");
  json.arrayBegin();
  for (const auto &device : devices)
    device.serializeToJSON(json);
  json.arrayEnd();
  json.attributeEnd();
  json.objectEnd();
}

std::string DeviceSpecCollection::serializeToJSON() const {
  std::string str;
  llvm::raw_string_ostream os(str);
  llvm::json::OStream json(os);
  serializeToJSON(json);
  return str;
}

ErrorOr<DeviceSpecCollection>
DeviceSpecCollection::deserializeFromJSON(const llvm::json::Value *json) {
  const llvm::json::Object *object = json->getAsObject();
  if (!object) {
    return Error(
        "ill-formed serialized DeviceSpecCollection: expecting object");
  }
  const llvm::json::Value *host = object->get("host");
  const llvm::json::Array *devices = object->getArray("devices");
  if (!host || !devices) {
    return Error(
        "ill-formed serialized DeviceSpecCollection: missing attributes");
  }
  auto hostOr = DeviceRef::deserializeFromJSON(host);
  if (hostOr) {
    return Error(Twine("ill-formed serialized DeviceSpecCollection: ") +
                 hostOr.getError());
  }

  DeviceSpecCollection result;
  result.host = std::move(*hostOr);

  for (const auto &v : *devices) {
    auto errOr = DeviceSpec::deserializeFromJSON(&v);
    if (errOr) {
      return Error(Twine("ill-formed serialized DeviceSpecCollection: ") +
                   errOr.getError());
    }
    result.devices.emplace_back(std::move(*errOr));
  }

  return result;
}

ErrorOr<DeviceSpecCollection>
DeviceSpecCollection::deserializeFromJSON(StringRef json) {
  llvm::Expected<llvm::json::Value> errOrValue = llvm::json::parse(json);
  if (llvm::Error err = errOrValue.takeError()) {
    return Error(Twine("ill-formed serialized target info: ") +
                 toString(std::move(err)));
  }
  return deserializeFromJSON(&errOrValue.get());
}

ErrorOr<const DeviceSpec *>
DeviceSpecCollection::findDeviceSpec(const DeviceRef &ref) const {
  auto itr = llvm::find_if(
      devices, [&ref](const DeviceSpec &device) { return device.ref == ref; });
  if (itr == devices.end()) {
    return Error(
        Twine("no such device spec for reference '" + ref.toString() + "'"));
  }
  return &(*itr);
}

const DeviceSpec &DeviceSpecCollection::getHostDeviceSpec() const {
  auto specOr = findDeviceSpec(host);
  assert(!specOr.isError() && "no such host device spec");
  return **specOr;
}

//===----------------------------------------------------------------------===//
// SIMD Width
//===----------------------------------------------------------------------===//

size_t M::simdWidthFromFeature(StringRef plainFeature) {
  if (plainFeature.contains("avx512"))
    return 512;
  if (plainFeature.contains("avx2"))
    return 256;
  // HVX states its vector length in bytes, so "length128b" is a 1024-bit
  // register. Without these two the Hexagon targets take the default below and
  // every width derived from the field is 8x short — which de-vectorizes the
  // kernel rather than miscomputing it, so nothing downstream complains.
  if (plainFeature.contains("hvx-length128b"))
    return 1024;
  if (plainFeature.contains("hvx-length64b"))
    return 512;
  return 128;
}

size_t M::simdWidthFromFeatures(StringRef llvmFeatureStr) {
  size_t width = 128;
  SmallVector<StringRef> tokens;
  llvmFeatureStr.split(tokens, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (StringRef token : tokens) {
    if (token.empty() || token[0] == '-')
      continue;
    width = std::max(width, simdWidthFromFeature(token.ltrim("+")));
  }
  return width;
}

size_t M::simdWidthFromFeatures(ArrayRef<std::string> plainFeatures) {
  size_t maxWidth = 128;
  for (StringRef plainFeature : plainFeatures) {
    maxWidth = std::max(maxWidth, simdWidthFromFeature(plainFeature));
    if (maxWidth == 512)
      return maxWidth;
  }
  return maxWidth;
}
