"""OpenAI `response_format` for structured outputs, on any provider.

`response_format` comes in two kinds (OpenAI Chat Completions):

    {"type": "json_object"}
    {"type": "json_schema", "json_schema": {"name": "...", "schema": {...}, "strict": true}}

`response_model_for(response_format)` turns either into a Pydantic model class
that the existing `StructuredOutputHandler` drives: a provider that can
constrain decoding (Ollama `format`, LM Studio / llama.cpp / OpenAI-compatible
`response_format`, MLX or Transformers with Outlines, Anthropic's forced tool)
receives the caller's schema verbatim (`model_json_schema()` returns it
unchanged); every other provider gets the schema in the prompt, and every
answer — constrained or not — is validated against the caller's schema
(`validate_instance`), with the handler's feedback retries on failure.

The validator covers the JSON Schema subset OpenAI structured outputs accept:
type (and type lists), properties, required, additionalProperties, items,
prefixItems, enum, const, anyOf/oneOf/allOf, $ref into $defs/definitions,
minItems/maxItems, minLength/maxLength, pattern, minimum/maximum,
exclusiveMinimum/exclusiveMaximum, multipleOf, minProperties/maxProperties.
`format` and annotation keywords are accepted and not enforced.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, model_validator

RESPONSE_FORMAT_TYPES = ("text", "json_object", "json_schema")
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_DEPTH = 64


class ResponseFormatError(ValueError):
    """The caller's response_format is not usable (a 400 for the caller)."""

    def __init__(self, message: str, param: str = "response_format") -> None:
        super().__init__(message)
        self.param = param


class SchemaViolation(ValueError):
    """An instance does not satisfy the schema; `path` names where."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path
        self.reason = message


def _type_ok(value: Any, kind: str) -> bool:
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer())
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    return True  # an unknown type name constrains nothing (refused at check time)


def _resolve(ref: str, root: Dict[str, Any]) -> Dict[str, Any]:
    if not ref.startswith("#"):
        raise SchemaViolation("", f"only local $ref is supported, not {ref!r}")
    node: Any = root
    for part in [p for p in ref[1:].split("/") if p]:
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise SchemaViolation("", f"$ref {ref!r} does not resolve")
        node = node[part]
    if not isinstance(node, dict):
        raise SchemaViolation("", f"$ref {ref!r} does not resolve to a schema")
    return node


def _at(path: str, key: Any) -> str:
    return f"{path}[{key}]" if isinstance(key, int) else (f"{path}.{key}" if path else str(key))


def _validate(value: Any, schema: Any, root: Dict[str, Any], path: str, depth: int) -> None:
    if depth > _MAX_DEPTH:
        raise SchemaViolation(path, "schema nesting is too deep")
    if schema is True or schema == {}:
        return
    if schema is False:
        raise SchemaViolation(path, "no value is allowed here")
    if not isinstance(schema, dict):
        return
    if "$ref" in schema:
        _validate(value, _resolve(str(schema["$ref"]), root), root, path, depth + 1)
    if "const" in schema and value != schema["const"]:
        raise SchemaViolation(path, f"must be {json.dumps(schema['const'])}")
    if "enum" in schema and isinstance(schema["enum"], list) and value not in schema["enum"]:
        raise SchemaViolation(path, f"must be one of {json.dumps(schema['enum'])}")
    kinds = schema.get("type")
    if kinds is not None:
        names = kinds if isinstance(kinds, list) else [kinds]
        if not any(_type_ok(value, str(k)) for k in names):
            raise SchemaViolation(path, f"must be of type {' or '.join(str(k) for k in names)}")
    for key in ("allOf",):
        for sub in schema.get(key) or []:
            _validate(value, sub, root, path, depth + 1)
    for key in ("anyOf", "oneOf"):
        subs = schema.get(key)
        if isinstance(subs, list) and subs:
            passed = 0
            first: Optional[SchemaViolation] = None
            for sub in subs:
                try:
                    _validate(value, sub, root, path, depth + 1)
                    passed += 1
                except SchemaViolation as exc:
                    first = first or exc
            if passed == 0:
                raise SchemaViolation(path, f"matches none of the {key} choices ({first.reason if first else ''})")
            if key == "oneOf" and passed > 1:
                raise SchemaViolation(path, "matches more than one oneOf choice")
    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for name in schema.get("required") or []:
            if name not in value:
                raise SchemaViolation(_at(path, name), "is required")
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            if name in props:
                _validate(item, props[name], root, _at(path, name), depth + 1)
            elif extra is False:
                raise SchemaViolation(_at(path, name), "is not allowed (additionalProperties is false)")
            elif isinstance(extra, dict):
                _validate(item, extra, root, _at(path, name), depth + 1)
        if isinstance(schema.get("minProperties"), int) and len(value) < schema["minProperties"]:
            raise SchemaViolation(path, f"needs at least {schema['minProperties']} properties")
        if isinstance(schema.get("maxProperties"), int) and len(value) > schema["maxProperties"]:
            raise SchemaViolation(path, f"allows at most {schema['maxProperties']} properties")
    elif isinstance(value, list):
        prefix = schema.get("prefixItems") if isinstance(schema.get("prefixItems"), list) else []
        for i, sub in enumerate(prefix[: len(value)]):
            _validate(value[i], sub, root, _at(path, i), depth + 1)
        items = schema.get("items")
        if items is not None and not isinstance(items, list):
            for i in range(len(prefix), len(value)):
                _validate(value[i], items, root, _at(path, i), depth + 1)
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            raise SchemaViolation(path, f"needs at least {schema['minItems']} items")
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            raise SchemaViolation(path, f"allows at most {schema['maxItems']} items")
    elif isinstance(value, str):
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
            raise SchemaViolation(path, f"must be at least {schema['minLength']} characters")
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            raise SchemaViolation(path, f"must be at most {schema['maxLength']} characters")
        if isinstance(schema.get("pattern"), str) and re.search(schema["pattern"], value) is None:
            raise SchemaViolation(path, f"must match the pattern {schema['pattern']!r}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(schema.get("minimum"), (int, float)) and value < schema["minimum"]:
            raise SchemaViolation(path, f"must be >= {schema['minimum']}")
        if isinstance(schema.get("maximum"), (int, float)) and value > schema["maximum"]:
            raise SchemaViolation(path, f"must be <= {schema['maximum']}")
        if isinstance(schema.get("exclusiveMinimum"), (int, float)) and value <= schema["exclusiveMinimum"]:
            raise SchemaViolation(path, f"must be > {schema['exclusiveMinimum']}")
        if isinstance(schema.get("exclusiveMaximum"), (int, float)) and value >= schema["exclusiveMaximum"]:
            raise SchemaViolation(path, f"must be < {schema['exclusiveMaximum']}")
        step = schema.get("multipleOf")
        if isinstance(step, (int, float)) and step > 0 and abs(value / step - round(value / step)) > 1e-9:
            raise SchemaViolation(path, f"must be a multiple of {step}")


def validate_instance(value: Any, schema: Dict[str, Any]) -> None:
    """Raises SchemaViolation when `value` does not satisfy `schema`."""
    _validate(value, schema, schema, "", 0)


_KNOWN_TYPES = {"null", "boolean", "integer", "number", "string", "array", "object"}


def _check_schema(node: Any, root: Dict[str, Any], path: str, depth: int) -> None:
    if depth > _MAX_DEPTH:
        raise ResponseFormatError(f"response_format.json_schema.schema nests deeper than {_MAX_DEPTH} levels.")
    if isinstance(node, bool):
        return
    if not isinstance(node, dict):
        raise ResponseFormatError(f"response_format.json_schema.schema{path} must be a JSON object.")
    kinds = node.get("type")
    if kinds is not None:
        names = kinds if isinstance(kinds, list) else [kinds]
        bad = [k for k in names if k not in _KNOWN_TYPES]
        if bad:
            raise ResponseFormatError(f"response_format.json_schema.schema{path}: unknown type {bad[0]!r}.")
    if "$ref" in node:
        try:
            _resolve(str(node["$ref"]), root)
        except SchemaViolation as exc:
            raise ResponseFormatError(f"response_format.json_schema.schema{path}: {exc.reason}.") from None
    if isinstance(node.get("pattern"), str):
        try:
            re.compile(node["pattern"])
        except re.error as exc:
            raise ResponseFormatError(f"response_format.json_schema.schema{path}: invalid pattern ({exc}).") from None
    for key in ("properties", "$defs", "definitions"):
        if isinstance(node.get(key), dict):
            for name, sub in node[key].items():
                _check_schema(sub, root, f"{path}.{key}.{name}", depth + 1)
    for key in ("items", "additionalProperties", "not"):
        if isinstance(node.get(key), dict):
            _check_schema(node[key], root, f"{path}.{key}", depth + 1)
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        if isinstance(node.get(key), list):
            for i, sub in enumerate(node[key]):
                _check_schema(sub, root, f"{path}.{key}[{i}]", depth + 1)


def parse_response_format(raw: Any) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """(kind, schema, name) for a response_format; raises ResponseFormatError.

    kind is "text", "json_object" or "json_schema"; schema is None unless
    kind is "json_schema"."""
    if raw is None:
        return "text", None, ""
    if not isinstance(raw, dict):
        raise ResponseFormatError("response_format must be an object such as {\"type\": \"json_object\"}.")
    kind = raw.get("type")
    if kind not in RESPONSE_FORMAT_TYPES:
        raise ResponseFormatError(
            f"response_format.type must be one of text, json_object, json_schema (got {kind!r}).",
            "response_format.type")
    if kind != "json_schema":
        return str(kind), None, ""
    spec = raw.get("json_schema")
    if not isinstance(spec, dict):
        raise ResponseFormatError("response_format.json_schema is required for type json_schema.",
                                  "response_format.json_schema")
    name = spec.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ResponseFormatError(
            "response_format.json_schema.name is required: letters, digits, underscores and dashes, up to 64.",
            "response_format.json_schema.name")
    schema = spec.get("schema")
    if not isinstance(schema, dict) or not schema:
        raise ResponseFormatError("response_format.json_schema.schema must be a JSON Schema object.",
                                  "response_format.json_schema.schema")
    root_type = schema.get("type")
    if root_type != "object" and not (isinstance(root_type, list) and "object" in root_type) \
            and not (root_type is None and isinstance(schema.get("properties"), dict)):
        raise ResponseFormatError("response_format.json_schema.schema must describe an object (\"type\": \"object\").",
                                  "response_format.json_schema.schema")
    _check_schema(schema, schema, "", 0)
    return "json_schema", schema, name


_OBJECT_SCHEMA: Dict[str, Any] = {"type": "object"}


def response_model_for(raw: Any) -> Optional[Type[BaseModel]]:
    """A Pydantic model class for `response_format`, or None for text.

    The model's JSON schema IS the caller's schema (providers that constrain
    decoding get it verbatim) and its validation IS `validate_instance`
    against that schema, so the structured-output handler's retries feed the
    exact violation back to the model."""
    kind, schema, name = parse_response_format(raw)
    if kind == "text":
        return None
    target: Dict[str, Any] = schema if schema is not None else dict(_OBJECT_SCHEMA)
    model_name = name if kind == "json_schema" else "json_object"

    class _ResponseFormatModel(BaseModel):
        model_config = ConfigDict(extra="allow")

        @model_validator(mode="before")
        @classmethod
        def _against_schema(cls, data: Any) -> Any:
            try:
                validate_instance(data, target)
            except SchemaViolation as exc:
                raise ValueError(f"does not match the response_format schema: {exc}") from None
            return data

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
            return json.loads(json.dumps(target))

    _ResponseFormatModel.__name__ = model_name
    _ResponseFormatModel.__qualname__ = model_name
    setattr(_ResponseFormatModel, "__abstractcore_response_format__", {"kind": kind, "schema": target, "name": model_name})
    return _ResponseFormatModel


def instance_of(result: Any) -> Dict[str, Any]:
    """The JSON object a validated `response_model_for` instance carries."""
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    if isinstance(result, dict):
        return result
    raise SchemaViolation("", "the structured output is not a JSON object")
