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
// This file defines common preprocessor methods.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_PREPROCESSOR_H
#define SUPPORT_PREPROCESSOR_H

#define STRINGIFY_IMPL(X) #X
#define STRINGIFY(X) STRINGIFY_IMPL(X)

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define M_UNIQUE_NAME(base) CONCAT(base, __COUNTER__)

// Macros to determine the number of arguments in __VA_ARGS__. For example if
// there's one argument in the __VA_ARGS__, it will end up taking the _1
// argument in VA_ARGS_COUNT_N, and push back the numbers, meaning N will end up
// being 1.
#define VA_ARGS_COUNT_(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, \
                       _14, _15, N, ...)                                       \
  N
#define VA_ARGS_COUNT(...)                                                     \
  VA_ARGS_COUNT_(__VA_ARGS__, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

// EXPAND_PARAMS will expand a list of types to a function signature,
// automatically naming the arguments, for example:
//
//   EXPAND_PARAMS(int, float, const char*)
//
// Will expand to:
//
//   int arg3, float arg2, const char* arg1
#define EXPAND_PARAMS_1(type) type arg1
#define EXPAND_PARAMS_2(type, ...) type arg2, EXPAND_PARAMS_1(__VA_ARGS__)
#define EXPAND_PARAMS_3(type, ...) type arg3, EXPAND_PARAMS_2(__VA_ARGS__)
#define EXPAND_PARAMS_4(type, ...) type arg4, EXPAND_PARAMS_3(__VA_ARGS__)
#define EXPAND_PARAMS_5(type, ...) type arg5, EXPAND_PARAMS_4(__VA_ARGS__)
#define EXPAND_PARAMS_6(type, ...) type arg6, EXPAND_PARAMS_5(__VA_ARGS__)
#define EXPAND_PARAMS_7(type, ...) type arg7, EXPAND_PARAMS_6(__VA_ARGS__)
#define EXPAND_PARAMS_8(type, ...) type arg8, EXPAND_PARAMS_7(__VA_ARGS__)
#define EXPAND_PARAMS_9(type, ...) type arg9, EXPAND_PARAMS_8(__VA_ARGS__)
#define EXPAND_PARAMS_10(type, ...) type arg10, EXPAND_PARAMS_9(__VA_ARGS__)
#define EXPAND_PARAMS_11(type, ...) type arg11, EXPAND_PARAMS_10(__VA_ARGS__)
#define EXPAND_PARAMS_12(type, ...) type arg12, EXPAND_PARAMS_11(__VA_ARGS__)
#define EXPAND_PARAMS_13(type, ...) type arg13, EXPAND_PARAMS_12(__VA_ARGS__)
#define EXPAND_PARAMS_14(type, ...) type arg14, EXPAND_PARAMS_13(__VA_ARGS__)
#define EXPAND_PARAMS_15(type, ...) type arg15, EXPAND_PARAMS_14(__VA_ARGS__)

#define EXPAND_PARAMS(...)                                                     \
  CONCAT(EXPAND_PARAMS_, VA_ARGS_COUNT(__VA_ARGS__))(__VA_ARGS__)

// EXPAND_ARGS will expand a list to numbered argument names, matching the
// names defined by EXPAND_PARAMS, for example:
//
//    EXPAND_ARGS(int, float, const char*)
//
// Will expand to:
//
//    arg3, arg2, arg1
#define EXPAND_ARGS_1(type) arg1
#define EXPAND_ARGS_2(type, ...) arg2, EXPAND_ARGS_1(__VA_ARGS__)
#define EXPAND_ARGS_3(type, ...) arg3, EXPAND_ARGS_2(__VA_ARGS__)
#define EXPAND_ARGS_4(type, ...) arg4, EXPAND_ARGS_3(__VA_ARGS__)
#define EXPAND_ARGS_5(type, ...) arg5, EXPAND_ARGS_4(__VA_ARGS__)
#define EXPAND_ARGS_6(type, ...) arg6, EXPAND_ARGS_5(__VA_ARGS__)
#define EXPAND_ARGS_7(type, ...) arg7, EXPAND_ARGS_6(__VA_ARGS__)
#define EXPAND_ARGS_8(type, ...) arg8, EXPAND_ARGS_7(__VA_ARGS__)
#define EXPAND_ARGS_9(type, ...) arg9, EXPAND_ARGS_8(__VA_ARGS__)
#define EXPAND_ARGS_10(type, ...) arg10, EXPAND_ARGS_9(__VA_ARGS__)
#define EXPAND_ARGS_11(type, ...) arg11, EXPAND_ARGS_10(__VA_ARGS__)
#define EXPAND_ARGS_12(type, ...) arg12, EXPAND_ARGS_11(__VA_ARGS__)
#define EXPAND_ARGS_13(type, ...) arg13, EXPAND_ARGS_12(__VA_ARGS__)
#define EXPAND_ARGS_14(type, ...) arg14, EXPAND_ARGS_13(__VA_ARGS__)
#define EXPAND_ARGS_15(type, ...) arg15, EXPAND_ARGS_14(__VA_ARGS__)

#define EXPAND_ARGS(...)                                                       \
  CONCAT(EXPAND_ARGS_, VA_ARGS_COUNT(__VA_ARGS__))(__VA_ARGS__)

#endif // SUPPORT_PREPROCESSOR_H
