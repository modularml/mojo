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

// Stands in for libibverbs in NixlPluginDirTest. Built twice against different
// version scripts to mirror the two copies that coexist in a serving image: the
// EFA-patched build we stage next to the plugins (exports the private symbols
// its matching libefa binds, at IBVERBS_PRIVATE_59) and the distro build that
// unrelated components can drag in first (public symbols only). Both carry the
// same SONAME, which is what makes them collide.
//
// The toolchain compiles with hidden visibility, which a version script cannot
// override, so the exported entry points say so explicitly.

#define FIXTURE_EXPORT extern "C" __attribute__((visibility("default")))

FIXTURE_EXPORT int ibv_fixture_open() { return 0; }

#if FIXTURE_HAS_IBVERBS_PRIVATE_59
FIXTURE_EXPORT int ibv_fixture_private() { return 0; }
#endif
