#!/bin/bash
##===----------------------------------------------------------------------===##
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##===----------------------------------------------------------------------===##

set -euo pipefail

readonly binary=Support/test/binary_id_binary
if [[ "$OSTYPE" == darwin* ]]; then
  library=Support/test/libbinary_id_shared_library.dylib
else
  library=Support/test/libbinary_id_shared_library.so
fi

output=$($binary)

binary_id=$(echo "$output" | grep "binary id: " | cut -d" " -f3)
library_id=$(echo "$output" | grep "shared library id: " | cut -d" " -f4)

if [[ -z "$binary_id" || -z "$library_id" ]]; then
  echo "error: binary and library id should be non empty: $binary_id, $library_id" >&2
  exit 1
fi

if [[ "$binary_id" == "$library_id" ]]; then
  echo "error: binary and library id should differ: $binary_id, $library_id" >&2
  exit 1
fi

if [[ "$OSTYPE" == darwin* ]]; then
  real_binary_id=$(dwarfdump -u "$binary" | tr '[:upper:]' '[:lower:]' | tr -d '-')
  if ! echo "$real_binary_id" | grep -q "uuid: $binary_id"; then
    echo "error: binary id '$binary_id' does not match real id: '$real_binary_id'" >&2
    exit 1
  fi

  real_library_id=$(dwarfdump -u "$library" | tr '[:upper:]' '[:lower:]' | tr -d '-')
  if ! echo "$real_library_id" | grep -q "uuid: $library_id"; then
    echo "error: library id '$library_id' does not match real id: '$real_library_id'" >&2
    exit 1
  fi
else
  readobj=../+llvm_configure+llvm-project/llvm/llvm-readobj
  real_binary_id=$("$readobj" -n "$binary")
  if ! echo "$real_binary_id" | grep -q "Build ID: $binary_id"; then
    echo "error: binary id '$binary_id' does not match real id: '$real_binary_id'" >&2
    exit 1
  fi

  real_library_id=$("$readobj" -n "$library")
  if ! echo "$real_library_id" | grep -q "Build ID: $library_id"; then
    echo "error: library id '$library_id' does not match real id: '$real_library_id'" >&2
    exit 1
  fi
fi
