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
// This file defines macros for exporting symbols.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_EXPORT_H
#define SUPPORT_EXPORT_H

#if (defined(_WIN32) || defined(__CYGWIN__))
#ifdef MODULAR_BUILDING_LIBRARY
#define MODULAR_VISIBILITY_EXPORT __declspec(dllexport)
#else
#define MODULAR_VISIBILITY_EXPORT __declspec(dllimport)
#endif
#else
#define MODULAR_VISIBILITY_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#define MODULAR_EXPORT extern "C" MODULAR_VISIBILITY_EXPORT
#else
#define MODULAR_EXPORT MODULAR_VISIBILITY_EXPORT
#endif

// We have to have a way to turn off exports when we're building as a static
// lib - MSVC doesn't allow dllimport/dllexport on static library functions.
#ifndef MODULAR_NO_EXPORT
#define MODULAR_CXX_EXPORT MODULAR_VISIBILITY_EXPORT
#else
#define MODULAR_CXX_EXPORT
#endif

// For CompilerRT we need the runtime entry points to have unmangled names,
// but currently do not wish to give them default visibility in any dylib
// they end up within.
#define COMPILERRT_EXPORT extern "C"

#if (defined(_WIN32) || defined(__CYGWIN__))
#ifdef MODULAR_BUILDING_COMPILERRT
#define COMPILERRT_VISIBILITY_EXPORT __declspec(dllexport)
#else
#define COMPILERRT_VISIBILITY_EXPORT __declspec(dllimport)
#endif
#else
#define COMPILERRT_VISIBILITY_EXPORT __attribute__((visibility("default")))
#endif

#if (defined(_WIN32) || defined(__CYGWIN__))
#ifdef MODULAR_BUILDING_DRIVER
#define DRIVER_VISIBILITY_EXPORT __declspec(dllexport)
#else
#define DRIVER_VISIBILITY_EXPORT __declspec(dllimport)
#endif
#else
#define DRIVER_VISIBILITY_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#define MODULAR_DRIVER_EXPORT extern "C" DRIVER_VISIBILITY_EXPORT
#else
#define MODULAR_DRIVER_EXPORT DRIVER_VISIBILITY_EXPORT
#endif

#endif // SUPPORT_EXPORT_H
