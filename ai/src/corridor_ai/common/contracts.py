"""Validation of messages against shared/contracts JSON Schemas."""

from __future__ import annotations

import json
from functools import cache
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from corridor_ai.common.paths import SCHEMA_DIR


@cache
def validator(schema_name: str) -> Draft202012Validator:
    schema = json.loads((SCHEMA_DIR / f"{schema_name}.schema.json").read_text("utf-8"))
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validation_errors(schema_name: str, message: dict[str, Any]) -> list[str]:
    return [f"{e.json_path}: {e.message}" for e in validator(schema_name).iter_errors(message)]


def validate(schema_name: str, message: dict[str, Any]) -> None:
    """Raise ValueError if `message` does not match the named schema."""
    errors = validation_errors(schema_name, message)
    if errors:
        raise ValueError(f"{schema_name} message invalid: {errors}")
