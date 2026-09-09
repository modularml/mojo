# Cyclopts Configuration Utilities

This module provides configuration utilities for Cyclopts CLI applications.

## Configuration Precedence

Configuration values are resolved in the following order (highest to lowest
priority):

1. **CLI arguments** - Values provided directly on the command line
2. **Environment variables** - Values from `MODULAR_*` environment variables
3. **Config files (YAML)** - Values from YAML configuration files specified via
   `--config-file`
4. **Defaults** - Default values defined in configuration classes

**Note:** Due to how cyclopts processes config sources, environment variables
are applied before Pydantic validation runs. This means config files cannot
override environment variables while still allowing CLI arguments to override
everything.

## Quick Start

### Basic Example

```python
from __future__ import annotations

from typing import Any

import yaml
from cyclopts import App, Parameter
from cyclopts.config import Env
from pydantic import BaseModel, Field, model_validator


# Define your subconfig classes.
class ModelConfig(ConfigFileModel):
    model: str | None = Field(default=None)
    seed: int = Field(default=42)


class ShapeConfig(ConfigFileModel):
    batch_size: int = Field(default=1)
    input_len: int = Field(default=256)


# Make sure to inherit from ConfigFileModel and include the @Parameter decorator
# with name="*".
@Parameter(name="*")
class MyCLIConfig(ConfigFileModel):
    model_config: ModelConfig = ModelConfig()
    shape_config: ShapeConfig = ShapeConfig()


def main() -> None:
    """Main entry point for the CLI application."""
    # Create the Cyclopts app with environment variable config source
    app = App(
        name="my_cli",
        help="My CLI application",
        help_formatter="plain",
        config=[
            Env(prefix="MODULAR_"),
        ],
    )

    # Define your command
    @app.default
    def run(config: MyCLIConfig = MyCLIConfig()) -> None:
        """Run the application."""
        print(f"Model: {config.model_config.model}")
        print(f"Batch size: {config.shape_config.batch_size}")

    app()


if __name__ == "__main__":
    main()
```

### Config File Example

Create a YAML config file (`config.yaml`):

```yaml
model: "llama-2-7b"
seed: 123
batch_size: 8
input_len: 512
```

Run your CLI:

```bash
# Using config file
python my_cli.py --config-file config.yaml

# CLI args override config file
python my_cli.py --config-file config.yaml --batch-size 16

# Environment variables override config file (due to cyclopts processing order)
export MODULAR_BATCH_SIZE=4
python my_cli.py --config-file config.yaml  # batch_size will be 4 from env var, not 8 from config
```

### Environment Variables Example

Run your CLI:

```bash
# Environment variables override defaults
export MODULAR_BATCH_SIZE=8
export MODULAR_MODEL=llama-2-7b
python my_cli.py

# CLI args override environment variables
python my_cli.py --batch-size 16  # batch_size will be 16, not 8 from env
```

## Usage Patterns

### Pattern 1: Config Files Only

```python
# All config classes inherit from ConfigFileModel
class ModelConfig(ConfigFileModel):
    model: str | None = Field(default=None)

@app.default
def run(model_config: ModelConfig | None = None) -> None:
    # Use model_config...

# config.yaml:
# model: llama-2-7b
# 
# Run: python my_cli.py --config-file config.yaml
```

### Pattern 2: Environment Variables Only

```python
app = App(
    name="my_cli",
    help="My CLI",
    help_formatter="plain",
    config=[Env(prefix="MODULAR_")],
)

@app.default
def run(model_config: ModelConfig | None = None) -> None:
    # Use model_config...

# Users can set: export MODULAR_MODEL=llama-2-7b
# Then run: python my_cli.py
```

## Precedence Examples

### Example 1: CLI Overrides Everything

```bash
# config.yaml has: batch_size: 8
# Environment has: MODULAR_BATCH_SIZE=4
# Default is: batch_size: 1

python my_cli.py --config-file config.yaml --batch-size 16
# Result: batch_size = 16 (from CLI)
```

### Example 2: Env Vars Override Config File

```bash
# config.yaml has: batch_size: 8
# Environment has: MODULAR_BATCH_SIZE=4
# Default is: batch_size: 1

python my_cli.py --config-file config.yaml
# Result: batch_size = 4 (from env var, not config file)
# Note: Due to cyclopts processing order, env vars override config files
```

### Example 3: Env Vars Override Defaults

```bash
# Environment has: MODULAR_BATCH_SIZE=4
# Default is: batch_size: 1

python my_cli.py
# Result: batch_size = 4 (from env var, not default)
```

### Example 4: Defaults Used When Nothing Else Provided

```bash
# No config file, no env vars
# Default is: batch_size: 1

python my_cli.py
# Result: batch_size = 1 (from default)
```

## Environment Variable Naming

Environment variables must be prefixed with `MODULAR_` and use uppercase
with underscores. The variable name is derived from the parameter name:

- Parameter: `batch_size` → Env var: `MODULAR_BATCH_SIZE`
- Parameter: `input_len` → Env var: `MODULAR_INPUT_LEN`
- Parameter: `model` → Env var: `MODULAR_MODEL`

## Config File Format

Config files should be YAML format. The structure should match your config
class fields:

```yaml
# config.yaml
model: "llama-2-7b"
seed: 123
batch_size: 8
input_len: 512
```

When using nested config classes, you can structure the YAML accordingly:

```yaml
# For nested configs like BenchmarkConfig with ModelConfig,
# ShapeConfig, etc.
model: "llama-2-7b"
seed: 123
batch_size: 8
input_len: 512
```

## Best Practices

1. **Always inherit from ConfigFileModel** for config classes that should
   support config files.
2. **Always provide defaults** in your configuration classes to ensure
   the app works without needing to specify any configuration.
3. **Use name="*" in @Parameter** Add the `@Parameter` decorator to the top
   level ConfigFileModel to unroll / flatten the variables / would be CLI flags
   within the singleton config class.
4. **Pass only one parent ConfigFileModel arg** to the `@app.default` decorated
   function.
5. **Use config files** for complex configurations or when you want to
   version control your settings across various workflows.
