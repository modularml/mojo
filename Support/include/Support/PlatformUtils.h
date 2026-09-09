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
// This file declares abstractions for platform specific macros.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PLATFORM_UTILS_H
#define SUPPORT_PLATFORM_UTILS_H

// clang-format off
// allow long #if lines for easier linting
#if defined(__x86_64__) || defined(__x86_64) || defined(_M_AMD64) || defined(_M_X64)
#define MODULAR_X86_64 1
#else
#define MODULAR_X86_64 0
#endif

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define MODULAR_ARM_NEON 1
#else
#define MODULAR_ARM_NEON 0
#endif

#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || defined(_M_ARM)
#define MODULAR_ARM 1
#else
#define MODULAR_ARM 0
#endif

#if defined(_WIN32) || defined(_WIN64)
#define MODULAR_LINUX 0
#define MODULAR_MACOS 0
#define MODULAR_WINDOWS 1
#elif defined(__APPLE__) || defined(__MACH__)
#define MODULAR_LINUX 0
#define MODULAR_MACOS 1
#define MODULAR_WINDOWS 0
#elif defined(__linux__)
#define MODULAR_LINUX 1
#define MODULAR_MACOS 0
#define MODULAR_WINDOWS 0
#else
#error "Could not determine platform"
#endif

#endif // SUPPORT_PLATFORM_UTILS_H
