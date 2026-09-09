#!/usr/bin/env python3
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

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import jinja2
from mojodoc_api_href import create_api_link, pad_backticks


def _configure_jinja_env(
    environment: jinja2.Environment,
    hosted_on_mojolang: bool,
    stability_doc_url: str,
) -> None:
    """Attach filters and ``api_link`` used by mojodoc templates."""

    environment.filters["pad_backticks"] = pad_backticks
    environment.globals["api_link"] = partial(
        create_api_link,
        hosted_on_mojolang=hosted_on_mojolang,
    )
    # A global, not a render variable: templates import macros.jinja without
    # context, so only globals are visible inside the stability_marker macro.
    environment.globals["stability_doc_url"] = stability_doc_url


def addStabilityMarker(mojo_json, mode: str) -> None:  # noqa: ANN001
    """Set stabilityMarker/stabilityMarkerVersion on each API declaration.

    Args:
        mojo_json: Module-level JSON declaration to annotate.
        mode: One of "all", "stable", or "none".
            - "none": no markers are set.
            - "stable": only stable APIs get a marker.
            - "all": stable APIs get "Stable"; everything else gets "Unstable".
    """
    if mode == "none":
        return

    def _apply(decl) -> None:  # noqa: ANN001
        if decl.get("isStable"):
            decl["showStabilityMarker"] = True
            since = decl.get("sinceVersion")
            # normalize version to three digits
            digits = since.split(".")
            for _ in range(3 - len(digits)):
                digits.append("0")
            decl["sinceVersion"] = ".".join(digits)
        elif mode == "all":
            decl["showStabilityMarker"] = True

    # Top-level aliases and functions
    for alias in mojo_json.get("aliases", []):
        _apply(alias)
    for overload_set in mojo_json.get("functions", []):
        for fn in overload_set.get("overloads", []):
            _apply(fn)

    # Structs and traits: the type itself, its aliases, and its methods
    for type_decl in mojo_json.get("structs", []) + mojo_json.get("traits", []):
        _apply(type_decl)
        for alias in type_decl.get("aliases", []):
            _apply(alias)
        for overload_set in type_decl.get("functions", []):
            for fn in overload_set.get("overloads", []):
                _apply(fn)


def addImplicitConversionDecorator(mojo_json) -> None:  # noqa: ANN001
    """Show @implicit on implicit constructors.  The isImplicitConversion
    flag should only appear on constructors, but check just in case.
    For now we assume that this is the only decorator we show
    (the @stable decorator is not in the templates at present)."""
    for struct in mojo_json["structs"] + mojo_json["traits"]:
        for overload_set in struct["functions"]:
            for function in overload_set["overloads"]:
                if function["isImplicitConversion"]:
                    if function["name"] == "__init__":
                        function["implicit"] = True
                    else:
                        print(
                            f"Error: {struct['name']}.{function['name']} "
                            + "declared with @implicit but is not a constructor.",
                            file=sys.stderr,
                        )
                        sys.exit(1)


def copyFieldTypesToValue(mojo_json) -> None:  # noqa: ANN001
    """The type of struct fields is expected by the doc generator to be in a
    field called 'value'."""
    for struct in mojo_json["structs"] + mojo_json["traits"]:
        for field in struct["fields"]:
            field["value"] = field["type"]


def processTraitMethods(mojo_json) -> None:  # noqa: ANN001
    """Dividing the single list of methods into required and provided lists,
    where provided methods are those with a default implementation.
    Note that a single function may have both required overloads and provided
    overloads."""
    for trait in mojo_json["traits"]:
        trait["required_methods"] = []
        trait["provided_methods"] = []

        for function in trait["functions"]:
            required_overloads = []
            provided_overloads = []

            for overload in function["overloads"]:
                if overload["hasDefaultImplementation"]:
                    provided_overloads.append(overload)
                else:
                    required_overloads.append(overload)

            if len(required_overloads) > 0:
                required_method = function.copy()
                required_method["overloads"] = required_overloads
                trait["required_methods"].append(required_method)
            if len(provided_overloads) > 0:
                provided_method = function.copy()
                provided_method["overloads"] = provided_overloads
                trait["provided_methods"].append(provided_method)


def removeSelfArgumentFromStructMethods(mojo_json) -> None:  # noqa: ANN001
    """If we are in a non-static struct method, we don't want to show the first argument (self) as an
    argument in the documentation. So we remove it from all the functions that are child of structs.
    """
    for struct in mojo_json["structs"] + mojo_json["traits"]:
        for overload_set in struct["functions"]:
            for function in overload_set["overloads"]:
                if (
                    not function["isStatic"]
                    and function["args"]
                    and function["args"][0]["type"] == "Self"
                    and function["args"][0]["name"] == "self"
                ):
                    function["args"].pop(0)


def removeArgumentsWithoutDocumentation(mojo_json) -> None:  # noqa: ANN001
    """We've been omitting function arguments without documentation from docstring, so we remove them
    from top-level functions and struct methods."""

    def process_decl_with_functions(decl) -> None:  # noqa: ANN001
        for overloadSet in decl["functions"]:
            for function in overloadSet["overloads"]:
                function["args"] = [
                    arg for arg in function["args"] if arg["description"]
                ]

    for struct in mojo_json["structs"] + mojo_json["traits"]:
        process_decl_with_functions(struct)
    process_decl_with_functions(mojo_json)


def removeParametersWithoutDocumentation(mojo_json) -> None:  # noqa: ANN001
    """We've been omitting parameters without documentation from docstring, so we remove them
    from top-level functions, struct methods and structs"""

    def process_decl_with_parameters(decl) -> None:  # noqa: ANN001
        decl["parameters"] = [
            param for param in decl["parameters"] if param["description"]
        ]

    def process_decl_with_functions(decl) -> None:  # noqa: ANN001
        for overloadSet in decl["functions"]:
            for function in overloadSet["overloads"]:
                process_decl_with_parameters(function)

    for struct in mojo_json["structs"]:
        process_decl_with_functions(struct)
        process_decl_with_parameters(struct)

    for trait in mojo_json["traits"]:
        process_decl_with_functions(trait)

    process_decl_with_functions(mojo_json)


def removeStaticFromInitializers(mojo_json) -> None:  # noqa: ANN001
    """Removes 'isStatic' from struct initializer functions.

    The "isStatic" attribute is set to `true` for all `FnOp` for which
    `isStatic` is true. This is confusing for readers of the documentation, who
    would see that the `__init__` method of a struct with value semantics (that
    is, one decorated with `@value`) is "static," but a similar `__init__`
    method for a struct not decorated as such it not "static."

    At a high level, the concern is that users will not always understand why
    the LIT dialect treats certain functions as static or not, and we munge the
    data here to be simpler."""

    def process_decl_with_functions(decl) -> None:  # noqa: ANN001
        for overloadSet in decl["functions"]:
            for function in overloadSet["overloads"]:
                name = function["name"]
                isInitializer = name.startswith("__") and name.endswith(
                    "init__"
                )
                function["isStatic"] = (
                    function["isStatic"] and not isInitializer
                )

    for struct in mojo_json["structs"] + mojo_json["traits"]:
        process_decl_with_functions(struct)
    process_decl_with_functions(mojo_json)


# The slug is the name of the module, except for the index
# module, which is named "index_" to avoid a name conflict with
# the index file.
def nameToSlug(name):  # noqa: ANN001, ANN201
    return "index_" if name == "index" else name


def generateMarkdown(
    mojo_json,  # noqa: ANN001
    version: str,
    output: Path,
    environment: jinja2.Environment,
    template: jinja2.Template,
    parent_json=None,  # noqa: ANN001
    is_nested=False,  # noqa: ANN001
    namespace=None,  # noqa: ANN001
    show_stability_markers: str = "none",
    docs_title: str | None = None,
) -> None:
    """Generate markdown docs from `mojo doc` JSON data.

    This function recursively processes Mojo documentation JSON data and generates
    corresponding markdown files using Jinja2 templates. It handles packages, modules,
    structs, traits, and functions, applying various transformations to the data
    before rendering.

    Args:
        mojo_json: The JSON data structure containing Mojo documentation information.
        version: The version string to be included in the generated documentation.
        output: The base output directory path where generated markdown files will be written.
        environment: The Jinja2 environment configured with template loaders and settings.
            Used to load and render documentation templates.
        template: The Jinja2 template to use for rendering the current JSON data.
        parent_json: The parent JSON data structure when processing nested elements.
            Used for context when generating documentation for child modules, structs, etc.
            Defaults to None.
        is_nested: Flag indicating whether this is a nested call within a package/module hierarchy.
            Affects path generation and namespace handling. Defaults to False.
        namespace: The current namespace path (dot-separated).
            Used to generate fully qualified names and proper cross-references.
            Defaults to None.
        docs_title: Custom title for the top-level package index page. Only
            applied at the root level; nested sub-packages use their own names.
            Defaults to None.
    """
    name = mojo_json["name"]

    # Add the module name to the JSON only if the parent is "__init__"
    # so we can create the proper "view source" link.
    if parent_json and parent_json["name"] == "__init__":
        mojo_json["module_name"] = parent_json["name"]

    # Skip private modules.
    if name != "__init__" and name.startswith("_"):
        return

    # If the json is a package, we recurse into the nested modules.
    if mojo_json["kind"] == "package":
        # If the package is nested, we need to add the package name to the
        # output path.
        if is_nested:
            output = output / name

        namespace = namespace + "." + name if namespace else name

        for module in mojo_json["modules"] + mojo_json["packages"]:
            generateMarkdown(
                module,
                version,
                output,
                environment,
                template,
                parent_json=mojo_json,
                is_nested=True,
                namespace=namespace,
                show_stability_markers=show_stability_markers,
                docs_title=docs_title if not is_nested else None,
            )
        return
    else:
        mojo_json["version"] = version
        mojo_json["slug"] = nameToSlug(mojo_json["name"])
        mojo_json["namespace"] = namespace

    # If its a module, we apply separate templates for struct/trait or function
    if mojo_json["kind"] == "module":
        addStabilityMarker(mojo_json, show_stability_markers)
        for transformation in [
            addImplicitConversionDecorator,
            copyFieldTypesToValue,
            processTraitMethods,
            removeParametersWithoutDocumentation,
            removeArgumentsWithoutDocumentation,
            removeSelfArgumentFromStructMethods,
            removeStaticFromInitializers,
        ]:
            transformation(mojo_json)

        # Use the member name as the `slug` for Docusaurus URLs (case sensitive).
        # But don't use the `__init__` name in the path. Normally this doesn't
        # matter, because the init module is just the index file and has no
        # descendant members. But in the event that the `__init__.mojo` file does
        # include code, we don't want the `__init__` name in the path.
        if name != "__init__":
            output = output / Path(mojo_json["slug"])
        struct_template = environment.get_template("mojodoc_struct.md")
        function_template = environment.get_template("mojodoc_function.md")

        namespace = namespace + "." + name if namespace else name

        # Save list of all struct names to compare to function names below
        struct_names = []
        for struct in mojo_json["structs"]:
            struct_names.append(struct["name"])
            generateMarkdown(
                struct,
                version,
                output,
                environment,
                struct_template,
                parent_json=mojo_json,
                is_nested=True,
                namespace=namespace,
                show_stability_markers=show_stability_markers,
            )

        for trait in mojo_json["traits"]:
            generateMarkdown(
                trait,
                version,
                output,
                environment,
                struct_template,
                parent_json=mojo_json,
                is_nested=True,
                namespace=namespace,
                show_stability_markers=show_stability_markers,
            )

        for function in mojo_json["functions"]:
            # Account for function names that match sibling struct names.
            # URL paths are case-sensitive but the macOS filesystem is not, so
            # create unique filenames for these functions so we don't clobber
            # files when building on mac. Also create unique filenames for any
            # functions called index or Index for similar reasons.
            function["filename"] = function["name"]
            if (
                function["name"].capitalize() in struct_names
                or function["name"].capitalize() == "Index"
            ):
                function["filename"] = function["name"] + "-function"
            generateMarkdown(
                function,
                version,
                output,
                environment,
                function_template,
                parent_json=mojo_json,
                is_nested=True,
                namespace=namespace,
                show_stability_markers=show_stability_markers,
            )

        # Handle the init module.
        if name == "__init__" and parent_json:
            mojo_json["module_name"] = name  # For the "view source" link.
            output = output / Path("index.md")

            # Add links to the public modules and packages in the parent.
            mojo_json["modules"] = [
                {
                    "name": module["name"],
                    "slug": nameToSlug(module["name"]),
                    "kind": "module_link",
                    "summary": module["summary"],
                }
                for module in parent_json["modules"]
                if not module["name"].startswith("_")
            ]
            mojo_json["packages"] = [
                {
                    "name": package["name"],
                    "kind": "package_link",
                    "summary": package["summary"],
                }
                for package in parent_json["packages"]
                if not package["name"].startswith("_")
            ]

            mojo_json["name"] = docs_title or parent_json["name"]
        else:
            output = output / Path("index.md")
        mojo_json["slug"] = " "
    elif mojo_json["kind"] in ("struct", "trait"):
        output = output / Path(name + ".md")
    elif mojo_json["kind"] == "function":
        # Account for function names that match sibling struct names
        name = mojo_json["filename"]
        mojo_json["slug"] = name
        output = output / Path(name + ".md")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as output_file:
        # The first line must start with front matter, not a lint comment
        markdown = template.render(decls=[mojo_json])
        lines = markdown.splitlines()
        if lines and "rumdl-disable" in lines[0]:
            lines.pop(0)
        output_file.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", help="the input Mojo documentation json file name"
    )
    parser.add_argument("-o", "--output", type=Path, help="the output path")
    parser.add_argument(
        "--show-stability-markers",
        choices=["all", "stable", "none"],
        default="none",
        help="Show stability markers: 'all' marks every API, "
        "'stable' marks only stable APIs, 'none' hides markers.",
    )
    parser.add_argument(
        "--docs-title",
        default=None,
        help="Custom title for the top-level package index page.",
    )
    parser.add_argument(
        "--stability-doc-url",
        default="",
        help="URL that stability markers link to. Markers are plain text "
        "when unset.",
    )
    parser.add_argument(
        "--hosted-on-mojolang",
        action="store_true",
        default=False,
        help=(
            "Generated docs are published on mojolang.org (stdlib/layout). "
            "Uses root-relative /docs/... links instead of absolute mojolang URLs."
        ),
    )
    args = parser.parse_args()

    with open(args.filename) as jsonFile:
        template_dir = os.path.join(
            os.path.dirname(__file__), "mojodoc-templates"
        )
        environment = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        _configure_jinja_env(
            environment,
            hosted_on_mojolang=args.hosted_on_mojolang,
            stability_doc_url=args.stability_doc_url,
        )
        template = environment.get_template("mojodoc_module.md")
        docJson = json.load(jsonFile)

        version = docJson["version"]
        decl = docJson["decl"]
        generateMarkdown(
            decl,
            version,
            args.output,
            environment,
            template,
            show_stability_markers=args.show_stability_markers,
            docs_title=args.docs_title,
        )
        # os.remove(args.filename)


if __name__ == "__main__":
    main()
