# ===----------------------------------------------------------------------=== #
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
# ===----------------------------------------------------------------------=== #


def process_record(id: Int) raises -> String:
    if id < 0:
        raise Error("invalid record ID: must be non-negative")
    if id > 999:
        raise Error("record not found")
    return String("record_", id)


def main():
    try:
        for id in [5, 0, 1001, -3, 42]:
            var result: String
            try:
                print()
                print("try     => id:", id)
                if id == 0:
                    continue
                result = process_record(id)
            except e:
                if "invalid" in String(e):
                    print("except  => fatal:", e)
                    raise e
                print("except  => handled:", e)
            else:
                print("else    => success:", result)
            finally:
                print("finally => done with id:", id)
    except e:
        print("\nre-raised error:", e)
