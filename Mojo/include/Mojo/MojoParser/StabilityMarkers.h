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
// This file declares utilities for stability markers (@stable decorator).
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_STABILITYMARKERS_H
#define KGEN_MOJOPARSER_STABILITYMARKERS_H

#include "llvm/ADT/StringRef.h"

namespace M::KGEN::LIT {

/// Returns true if the given package name has opted into stability tracking.
/// Packages that opt in treat symbols without @stable as unstable APIs.
/// Currently only "std" is opted in; "test_std_mock" is a test-only stand-in.
///
/// This is the single source of truth for the opted-in set. The same check
/// is used by the compiler (for unstable-API warnings) and by the doc
/// generator (to decide whether to show stability labels).
bool isPackageOptedIntoStabilityMarkers(llvm::StringRef packageName);

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_STABILITYMARKERS_H
