import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from .schemas import PipelineConfig


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ConfigValidationError(ValueError):
    """
    Raised when the YAML configuration file fails Pydantic validation.

    Wraps :class:`pydantic.ValidationError` and prepends the config file
    path so error messages are immediately actionable.

    Attributes
    ----------
    path : str
        Absolute path to the config file that triggered the error.
    original : pydantic.ValidationError
        The underlying Pydantic validation error.
    """

    def __init__(self, path: str, original: ValidationError) -> None:
        self.path = path
        self.original = original
        error_count = original.error_count()
        errors_detail = original.errors(include_url=False)
        summary_lines = [
            f"Configuration file '{path}' has {error_count} validation "
            f"error{'s' if error_count != 1 else ''}:\n"
        ]
        for err in errors_detail:
            loc = " -> ".join(str(p) for p in err["loc"])
            summary_lines.append(f"  [{loc}]  {err['msg']}  (input: {err.get('input')!r})")
        super().__init__("\n".join(summary_lines))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_yaml_omegaconf(path: str) -> dict[str, Any]:
    """
    Load a YAML file with OmegaConf, resolving ``${...}`` interpolations.

    Parameters
    ----------
    path : str
        Absolute or relative path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Plain Python dictionary (OmegaConf internals stripped).

    Raises
    ------
    ImportError
        If OmegaConf is not installed.
    FileNotFoundError
        If the file does not exist.
    """
    try:
        from omegaconf import OmegaConf  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "OmegaConf is not installed. "
            "Install it with: pip install omegaconf>=2.3.0"
        ) from exc

    cfg = OmegaConf.load(path)
    # to_container resolves all ${} interpolations and returns a plain dict/list
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[return-value]


def _load_yaml_pyyaml(path: str) -> dict[str, Any]:
    """
    Fallback loader using plain PyYAML (no interpolation support).

    Parameters
    ----------
    path : str
        Absolute or relative path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Raw parsed YAML as a Python dictionary.
    """
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TypeError(
            f"Expected the YAML file '{path}' to contain a mapping at the top "
            f"level, but got {type(data).__name__}."
        )
    return data


def _read_raw(path: str) -> dict[str, Any]:
    """
    Read a YAML file, preferring OmegaConf and falling back to PyYAML.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration as a plain Python dict.
    """
    try:
        raw = _load_yaml_omegaconf(path)
        logger.debug("Loaded '%s' with OmegaConf (interpolation supported).", path)
    except ImportError:
        logger.warning(
            "OmegaConf not found; falling back to PyYAML "
            "('${...}' interpolations will NOT be resolved)."
        )
        raw = _load_yaml_pyyaml(path)

    return raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> PipelineConfig:
    """

    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: '{path.resolve()}'"
        )

    path_str = str(path)
    logger.info("Loading configuration from '%s'.", path_str)

    raw: dict[str, Any] = _read_raw(path_str)

    try:
        config = PipelineConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigValidationError(path_str, exc) from exc

    logger.info(
        "Configuration validated: model=%s, country=%s, feature=%s",
        config.GENERAL_INFO.photos_dir,
        config.MODEL.backend ,
    )
    return config


def load_config_from_dict(data: dict[str, Any]) -> PipelineConfig:
    """
    Construct a :class:`PipelineConfig` directly from a Python dictionary.
    """
    try:
        return PipelineConfig.model_validate(data)
    except ValidationError as exc:
        # Use "<dict>" as the pseudo-path for in-memory configs
        raise ConfigValidationError("<dict>", exc) from exc


def dump_schema(indent: int = 2) -> str:
    """
    Return the JSON Schema for :class:`SimulationConfig` as a formatted string.

    Useful for generating documentation or validating configs with external
    tools (e.g. VS Code YAML extension, ``ajv``).

    Parameters
    ----------
    indent : int
        JSON indentation level.  Defaults to 2.

    Returns
    -------
    str
        Pretty-printed JSON Schema.

    Examples
    --------
    >>> from ag_cube_cm.config.loader import dump_schema
    >>> print(dump_schema())
    {
      "$defs": { ... },
      "properties": { ... },
      ...
    }
    """
    import json

    schema = PipelineConfig.model_json_schema()
    return json.dumps(schema, indent=indent)