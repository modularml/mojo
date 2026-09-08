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
"""Password database lookups for user account information.

The `pwd` package provides access to the Unix password database for retrieving
user account information. It offers a portable interface to query user records
by username or user ID on Unix-like systems. This package enables programs to
look up user details such as home directories, shells, and group memberships.

Use this package when you need to resolve user IDs to usernames, retrieve user
home directories, validate user existence, or access other user account
metadata on Unix-like systems.
"""

from .pwd import getpwnam, getpwuid
