#!/usr/bin/env python3
"""Validate shared/contracts.

Checks:
1. every JSON Schema in shared/contracts/schemas is a valid draft 2020-12 schema;
2. every example in schemas/examples/<name>.example.json validates against
   schemas/<name>.schema.json;
3. every `intersection_id`-style enum matches the ids in intersections.yaml.

Requires: jsonschema, pyyaml.  Usage: python scripts/validate_contracts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "shared" / "contracts"
SCHEMAS = CONTRACTS / "schemas"
EXAMPLES = SCHEMAS / "examples"


def _intersection_enums(node, path=""):
    """Yield (path, enum) for every enum that lists intersection ids."""
    if isinstance(node, dict):
        if "enum" in node and "lozan" in node["enum"]:
            yield path, node["enum"]
        for key, value in node.items():
            yield from _intersection_enums(value, f"{path}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _intersection_enums(value, f"{path}/{i}")


def main() -> int:
    errors: list[str] = []
    registry = yaml.safe_load((CONTRACTS / "intersections.yaml").read_text(encoding="utf-8"))
    ids = [item["id"] for item in registry["intersections"]]
    if len(ids) != len(set(ids)):
        errors.append("intersections.yaml: duplicate ids")

    schema_files = sorted(SCHEMAS.glob("*.schema.json"))
    if not schema_files:
        errors.append("no schemas found")

    for schema_path in schema_files:
        name = schema_path.name.removesuffix(".schema.json")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as exc:  # noqa: BLE001 - report any schema error
            errors.append(f"{schema_path.name}: invalid schema: {exc}")
            continue

        for enum_path, enum in _intersection_enums(schema):
            if list(enum) != ids:
                errors.append(f"{schema_path.name}{enum_path}: enum {enum} != intersections.yaml ids {ids}")

        example_path = EXAMPLES / f"{name}.example.json"
        if not example_path.exists():
            errors.append(f"{schema_path.name}: missing example {example_path.name}")
            continue
        example = json.loads(example_path.read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        for err in validator.iter_errors(example):
            errors.append(f"{example_path.name}: {err.json_path}: {err.message}")

        print(f"ok  {schema_path.name}")

    if errors:
        print("\nContract validation FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(f"\nAll {len(schema_files)} contracts valid; intersection ids: {ids}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
