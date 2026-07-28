"""
utils/config_loader.py
------------------------
Loads the YAML configs that drive training/inference: which variables to
use, where they live on disk, how to normalize them, and what spatial
region to train on. Adding a new variable means adding a config entry --
no code changes anywhere else.

Supports two conveniences from the example configs:
  - `!env "${VAR_NAME}"` tags, resolved from environment variables at load
    time (falls back to the literal string if the env var isn't set, with
    a warning, rather than silently substituting nothing).
  - `extends: other_file.yml` at the top level, for a base config a more
    specific one can build on. One level only (no deep chains) -- keeps
    this predictable rather than needing to trace a long inheritance tree.
"""
import os
import warnings

import yaml


class _EnvTagLoader(yaml.SafeLoader):
    pass


def _env_constructor(loader, node):
    value = loader.construct_scalar(node)
    if "${" in value and value.startswith("${") and value.endswith("}"):
        var_name = value[2:-1]
        resolved = os.environ.get(var_name)
        if resolved is None:
            warnings.warn(
                f"!env reference '${{{var_name}}}' has no matching environment "
                f"variable set -- using the literal string as a fallback. "
                f"Set it with: export {var_name}=/your/path"
            )
            return value
        return resolved
    return value


_EnvTagLoader.add_constructor("!env", _env_constructor)


def load_config(path):
    """Loads a YAML config, resolving !env tags and one level of `extends`."""
    path = os.path.abspath(path)
    with open(path) as f:
        config = yaml.load(f, Loader=_EnvTagLoader)

    if config is None:
        config = {}

    if "extends" in config:
        base_path = os.path.join(os.path.dirname(path), config["extends"])
        with open(base_path) as f:
            base_config = yaml.load(f, Loader=_EnvTagLoader) or {}
        merged = _deep_merge(base_config, config)
        merged.pop("extends", None)
        return merged

    return config


def _deep_merge(base, override):
    """Override values win; nested dicts merge recursively rather than
    replacing wholesale, so a specific config can tweak one field of a
    nested block without having to repeat the rest of it."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
