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

from mojodoc_api_href import (
    MAX_MOJO_ORIGIN,
    MOJOLANG_ORIGIN,
    create_api_link,
    resolve_api_href,
)


def test_std_path_kernel_tarball_uses_absolute_mojolang() -> None:
    assert resolve_api_href(
        "/std/collections/List",
        hosted_on_mojolang=False,
    ) == (f"{MOJOLANG_ORIGIN}/docs/std/collections/List")


def test_std_path_mojolang_tarball_root_relative() -> None:
    assert resolve_api_href(
        "/std/collections/List",
        hosted_on_mojolang=True,
    ) == ("/docs/std/collections/List")


def test_std_alias_fragment_kernel_tarball() -> None:
    assert resolve_api_href(
        "/std/builtin/#mutorigin",
        hosted_on_mojolang=False,
    ) == (f"{MOJOLANG_ORIGIN}/docs/std/builtin/#mutorigin")


def test_kernels_layout_kernel_tarball_root_relative() -> None:
    assert resolve_api_href(
        "/kernels/layout/tile_layout/TensorLayout",
        hosted_on_mojolang=False,
    ) == ("/api/mojo/layout/tile_layout/TensorLayout")


def test_kernels_layout_mojolang_tarball_uses_absolute_max_mojo() -> None:
    assert resolve_api_href(
        "/kernels/layout/tile_layout/TensorLayout",
        hosted_on_mojolang=True,
    ) == (f"{MAX_MOJO_ORIGIN}/api/mojo/layout/tile_layout/TensorLayout")


def test_kernels_layout_exact_module() -> None:
    assert resolve_api_href(
        "/kernels/layout",
        hosted_on_mojolang=False,
    ) == ("/api/mojo/layout")


def test_kernels_other_kernel_tarball_root_relative() -> None:
    assert resolve_api_href(
        "/kernels/linalg/foo/Bar",
        hosted_on_mojolang=False,
    ) == ("/api/mojo/linalg/foo/Bar")


def test_kernels_other_mojolang_tarball_uses_absolute_max_mojo() -> None:
    """Kernel cross-links from mojolang-hosted Markdown must be absolute."""
    assert resolve_api_href(
        "/kernels/linalg/foo/Bar",
        hosted_on_mojolang=True,
    ) == (f"{MAX_MOJO_ORIGIN}/api/mojo/linalg/foo/Bar")


def test_layout_path_mojolang_tarball_uses_absolute_max_mojo() -> None:
    assert resolve_api_href(
        "/layout/tile_layout/X",
        hosted_on_mojolang=True,
    ) == (f"{MAX_MOJO_ORIGIN}/api/mojo/layout/tile_layout/X")


def test_extensibility_tensor_kernel_tarball_root_relative() -> None:
    assert resolve_api_href(
        "/extensibility/tensor/foo/Bar",
        hosted_on_mojolang=False,
    ) == ("/api/mojo/extensibility/tensor/foo/Bar")


def test_extensibility_tensor_mojolang_tarball_uses_absolute_max_mojo() -> None:
    assert resolve_api_href(
        "/extensibility/tensor/foo/Bar",
        hosted_on_mojolang=True,
    ) == (f"{MAX_MOJO_ORIGIN}/api/mojo/extensibility/tensor/foo/Bar")


def test_mojo_max_package_root_tarball_root_relative() -> None:
    assert resolve_api_href(
        "/mojo/max",
        hosted_on_mojolang=False,
    ) == ("/api/mojo/max")


def test_mojo_max_subpackage_retains_max_segment() -> None:
    assert resolve_api_href(
        "/mojo/max/gpu/compute/mma",
        hosted_on_mojolang=False,
    ) == ("/api/mojo/max/gpu/compute/mma")


def test_empty_path() -> None:
    assert resolve_api_href("") == ""
    assert resolve_api_href(None) == ""


def test_fragment_only_after_split() -> None:
    assert resolve_api_href("#frag", hosted_on_mojolang=True) == "#frag"


def test_normalize_missing_leading_slash() -> None:
    assert resolve_api_href(
        "std/builtin/Int",
        hosted_on_mojolang=False,
    ) == (f"{MOJOLANG_ORIGIN}/docs/std/builtin/Int")


def test_private_module_path_returns_empty_href() -> None:
    assert (
        resolve_api_href(
            "/docs/std/collections/dict/_DictValueIter",
            hosted_on_mojolang=True,
        )
        == ""
    )


def test_private_symbol_in_public_module_returns_empty_href() -> None:
    assert (
        resolve_api_href(
            "/docs/std/python/_cpython/PyObjectPtr",
            hosted_on_mojolang=False,
        )
        == ""
    )


def test_public_std_path_still_links() -> None:
    assert (
        resolve_api_href(
            "/std/traits/movable/Movable",
            hosted_on_mojolang=True,
        )
        == "/docs/std/traits/movable/Movable"
    )


def test_private_path_renders_plain_type_markup() -> None:
    assert (
        create_api_link(
            "_DictValueIter",
            "/std/collections/dict/_DictValueIter",
            hosted_on_mojolang=True,
        )
        == "`_DictValueIter`"
    )


def test_public_path_renders_markdown_link() -> None:
    assert (
        create_api_link(
            "Movable",
            "/std/traits/movable/Movable",
            hosted_on_mojolang=True,
        )
        == "[`Movable`](/docs/std/traits/movable/Movable)"
    )


def test_padding_plain_when_private() -> None:
    assert (
        create_api_link(
            "List[Int]",
            "/std/collections/_Private",
            padding=True,
            hosted_on_mojolang=True,
        )
        == "``List[Int]``"
    )


def test_missing_path_renders_plain_type() -> None:
    assert create_api_link("Int", None, hosted_on_mojolang=True) == "`Int`"
