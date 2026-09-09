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

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   tests/data/preview_310/remove_newline_after_match.py
#
# ===----------------------------------------------------------------------=== #


def http_status(status):

    match status:

        case 400:

            return "Bad request"

        case 401:

            return "Unauthorized"

        case 403:

            return "Forbidden"

        case 404:

            return "Not found"


# output
def http_status(status):
    match status:
        case 400:
            return "Bad request"

        case 401:
            return "Unauthorized"

        case 403:
            return "Forbidden"

        case 404:
            return "Not found"
