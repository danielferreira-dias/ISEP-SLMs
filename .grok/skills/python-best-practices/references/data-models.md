# Data-model examples (Python 3.12+)

Use the decision list in `SKILL.md`. This file only shows the shape of each option.

## Pydantic v2 — untrusted input

Use `BaseModel` at HTTP, CLI, env, file, and LLM-output boundaries. Validate once, then pass
domain objects inward.

```python
from pydantic import BaseModel, Field, HttpUrl

class DiagnosisRequest(BaseModel):
    """Payload accepted from the evaluation API."""

    image_id: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)
    source: HttpUrl | None = None

    model_config = {"frozen": True, "extra": "forbid"}


request = DiagnosisRequest.model_validate(raw_json)
payload = request.model_dump(mode="json")
```

- v2 API only: `model_validate`, `model_dump`, `model_config`. Not `parse_obj`, `.dict()`,
  or inner `class Config`.
- `frozen=True` for value-like request/response objects.
- `extra = "forbid"` on external payloads so unknown fields fail loudly.
- Settings/env: `pydantic-settings.BaseSettings`, not a homemade `os.environ` parser.
- Do not use Pydantic for hot inner-loop tensors, numpy arrays, or trusted in-memory
  records. Convert at the edge.

## Dataclass — trusted internal records

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(slots=True, kw_only=True, frozen=True)
class Sample:
    """One labeled image after local preprocessing."""

    image_id: str
    path: Path
    diagnosis: str
    tags: tuple[str, ...] = ()


@dataclass(slots=True, kw_only=True)
class TrainState:
    """Mutable trainer counters. Not a validation boundary."""

    step: int = 0
    seen_ids: set[str] = field(default_factory=set)
```

- `slots=True` saves memory and catches accidental attributes.
- `kw_only=True` prevents positional call-site mistakes.
- `frozen=True` when the record should be hashable / shared.
- Never `def f(items: list[str] = [])`. Use `field(default_factory=list)`.

## TypedDict — dicts that must stay dicts

```python
from typing import NotRequired, TypedDict

class HFDatasetRow(TypedDict):
    """Row shape written to a Hugging Face parquet export."""

    image_id: str
    diagnosis: str
    split: str
    caption: NotRequired[str]
```

Access with `row["image_id"]`, not `row.image_id`. TypedDict is a static contract over
`dict`; it does no runtime validation. If you need validation, parse into Pydantic
(optionally `model_validate` from the TypedDict).

Use `total=False` only when most keys are optional; otherwise mark individual
`NotRequired` keys.

## NamedTuple — tiny immutable positional records

```python
from typing import NamedTuple

class Point(NamedTuple):
    """Pixel coordinate in image space."""

    y: int
    x: int
```

Good for unpacking (`y, x = point`) and as dict keys. If you add defaults, methods, or
validation, move to a frozen dataclass or Pydantic.

## Protocol — behavior, not data

```python
from typing import Protocol

class ImageStore(Protocol):
    """Anything that can load raw image bytes by id."""

    def get_bytes(self, image_id: str) -> bytes: ...
```

Depend on the protocol in function signatures. Do not create a Protocol for a bag of
fields — that is a dataclass or TypedDict.

## StrEnum — closed vocabularies

```python
from enum import StrEnum

class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
```

Use the enum in APIs (`split: Split`) instead of `Literal["train", "val"]` once the
set is reused in more than one module. `Literal` is fine for a one-off parameter.

## Conversion at the boundary

```python
def accept_request(raw: dict[str, object]) -> Sample:
    """Validate wire input, then return an internal dataclass."""

    parsed = DiagnosisRequest.model_validate(raw)
    return Sample(
        image_id=parsed.image_id,
        path=resolve_image(parsed.image_id),
        diagnosis="",
        tags=(),
    )
```

Keep Pydantic at the edge. Domain logic should see `Sample`, not `DiagnosisRequest`.
