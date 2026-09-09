# InstallMAX component

The `InstallMAX` component is a reusable MDX component that generates
installation instructions for the MAX package across multiple package
managers (pixi, uv, pip, and conda).

## Usage

Import and use the component in your MDX files:

```python
import InstallMAX from '../_includes/install-modular.mdx';

<InstallMAX folder="my-project" extraLibraries={["torch", "transformers"]} />
```

> [!CAUTION]
> You must use a relative path in the import statement so the page always
> uses the appropriate version of the included file based on the current docs
> version (from within the Docusaurus `versioned_docs/` path)

## Properties

- `folder` (optional): The name of the project folder to create during
  installation.
  - Type: `string`
  - Default: `"example-project"`

- `extraLibraries` (optional): Additional libraries to install alongside
  `max`.
  - Type: `string[]`
  - Default: `[]` | `max`

## Examples

Basic usage (MAX only):

```markdown
<InstallMAX />
```

 Custom project folder:

```markdown
<InstallMAX folder="llama-tutorial" />
```

With additional libraries:

```markdown
<InstallMAX folder="ml-project" extraLibraries={["torch", "numpy", "pandas"]} />
```

## Generated output

The component generates tabbed installation instructions for:

- **pixi**: Uses conda channels with `pixi add`
- **uv**: Uses custom index URLs via `uv add`
- **pip**: Standard pip installation with extra index URLs
- **conda**: Uses conda-forge and Modular's conda channels
