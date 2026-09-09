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

#include "Support/Profiling/ExternalProfilerAnnotation.h"

namespace M::Profiling::Detail {

// Weak no-integration default; a host-side external profiler annotation
// integration overrides it with a strong definition (see the header).
// `weak` (not a config define) so a single build of this library serves
// both kinds of binaries and the wiring is purely link-time.
__attribute__((weak)) const ExternalProfilerAnnotationSink *
acquireExternalProfilerAnnotationSink() {
  return nullptr;
}

} // namespace M::Profiling::Detail
