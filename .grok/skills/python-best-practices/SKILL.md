---
name: python-best-practices
description: >
  Apply Python 3.12+ best practices when writing, reviewing, or refactoring Python.
  Enforces readable spacing (never cramped), Google-style docstrings, modern type
  hints, and the right data model (Pydantic vs dataclass vs TypedDict vs NamedTuple).
  Also covers PEP 8/257, uv, Ruff, pytest, security, and packaging. Use when the user
  asks for Python best practices, boas práticas de Python, Pythonic code, PEP 8,
  docstrings, Pydantic, dataclasses, TypedDict, ruff, uv, type hints, pytest, code
  review of Python, or runs /python-best-practices.
---

# Python 3.12+ best practices

Apply this skill whenever writing, reviewing, or refactoring Python. Produce code a
human can scan; do not lecture unless the user asked for an explanation.

Target is **Python 3.12 and newer**. Do not emit 3.11-or-older typing syntax in new
code. If the surrounding file is older, modernize the code you touch; do not rewrite
the whole tree unless asked.

If the repo already has `pyproject.toml` / Ruff / pytest config, obey that config for
line length, quote style, and test layout. The rules below fill gaps.

## Readability (hard rule)

Code must not be agglomerated. Prefer a few extra lines over dense one-liners.

- Two blank lines between top-level functions and classes; one blank line between
  methods; a blank line between logical steps *inside* a function.
- One statement per line. No `;` chaining. No `if x: y` on one line except a
  one-branch `if` that is already the project style.
- Names over nested expressions: extract locals (`allowed_ids = set(ids)`) instead of
  stuffing a comprehension inside another.
- Guard clauses to keep nesting at most two or three levels.
- Functions do one thing. If a body needs comments to mark sections, split it.
- Do not pack an entire dict/list/call with many fields onto one line; break after the
  opening delimiter and trailing-comma the last item.
- Imports: stdlib, third-party, local — three groups, one blank line between groups.
  No unused imports. No `from module import *`.
- Naming: `snake_case` functions/variables, `PascalCase` types, `UPPER_SNAKE` constants,
  `_private` for internal. Names describe meaning, not type (`patients`, not `data` /
  `list1`).

```python
# Bad: cramped, unnamed, untyped
def run(p,d):
    return [x for x in p if x.id in {i for i in d}]

# Good
def select_patients(
    patients: Sequence[Patient],
    ids: Sequence[str],
) -> list[Patient]:
    """Return patients whose ids appear in `ids`."""

    allowed_ids = set(ids)
    return [patient for patient in patients if patient.id in allowed_ids]
```

## Types

Annotate public functions, dataclass/Pydantic fields, and non-obvious locals.

| Do | Do not |
| --- | --- |
| `list[str]`, `dict[str, int]`, `set[int]` | `List[str]`, `Dict[str, int]` |
| `str \| None`, `int \| str` | `Optional[str]`, `Union[int, str]` |
| `type PatientId = str` | `PatientId = NewType(...)` unless you need distinct nominal types |
| `class Box[T]: ...` / `def first[T](xs: list[T]) -> T` | `TypeVar` + `Generic[T]` in new code |
| `collections.abc.Sequence`, `Mapping`, `Callable` | `typing.List`, `typing.Dict`, `typing.Callable` |
| `pathlib.Path` | stringly paths in new APIs |
| `from typing import override` on overrides | undocumented silent overrides |

Keep `typing` for `TypedDict`, `NotRequired`, `Literal`, `Final`, `Protocol`, `Self`,
`Never`, and `ClassVar`. Do not add `from __future__ import annotations` unless the
file still needs it.

## Data models

Pick **one** shape per type. Full examples: [references/data-models.md](references/data-models.md).

1. **Untrusted input** (HTTP/JSON, CLI, env, files, LLM output) → **Pydantic v2**
   `BaseModel`. Validate at the boundary; do not re-validate the same object deeper in.
2. **Internal domain records** (trusted, in-process) → **`@dataclass`**. Prefer
   `@dataclass(slots=True, kw_only=True)`. Add `frozen=True` when the value should
   not mutate.
3. **Must stay a `dict`** (JSON-shaped payloads, `**kwargs` mapping, interop with
   untyped APIs) → **`TypedDict`**. Use `Required` / `NotRequired`. Never instantiate
   a TypedDict as if it were a class with attributes.
4. **Tiny immutable positional record** (unpacking, hashing, no validation) →
   **`NamedTuple`**. If it grows methods, defaults, or validation, switch to a frozen
   dataclass or Pydantic.
5. **Closed set of names** → **`enum.StrEnum`** / `IntEnum`. Not raw strings.
6. **Behavior contract** (duck typing) → **`Protocol`**. Not an ABC unless you need
   registration or shared implementation.
7. **Throwaway mapping** with no schema → plain `dict`. Do not use this at API
   boundaries.

Do not wrap every dict in Pydantic. Do not use a dataclass for request bodies. Do not
invent a custom `__init__` just to store fields.

Mutable defaults are bugs: use `field(default_factory=list)` / `Field(default_factory=list)`,
never `def f(items=[])`.

## Docstrings

Google style. Triple double-quotes. Imperative one-liner first
(`"""Return the cached tokenizer."""`), not `"""Returns..."` or a restatement of the
name.

Required on public modules, classes, and functions/methods that other modules call.
Optional on tiny private helpers whose name and types already say everything; if the
logic is non-obvious, write a docstring or a short comment *why*, not *what*.

- After the summary, a blank line, then details.
- `Args:` / `Returns:` / `Raises:` / `Yields:` / `Examples:` only when types alone
  are not enough (units, invariants, empty-input behavior, side effects).
- Do not duplicate type information already in annotations.
- Closing `"""` of a multi-line docstring sits on its own line.
- Keep docstrings in sync when behavior changes.

```python
def load_split(path: Path, split: Literal["train", "val"]) -> list[Sample]:
    """Load one dataset split from a parquet file.

    Args:
        path: Parquet file produced by the training export.
        split: Which split to keep. Other rows are ignored.

    Returns:
        Samples in file order. Empty if the split has no rows.

    Raises:
        FileNotFoundError: If `path` does not exist.
    """
```

Module docstring is the first statement in the file: what the module is for, not a
file-name echo.

## Language and stdlib

- `with` for files, locks, connections, temp dirs. No manual `close()` in a `finally`
  if a context manager exists.
- `pathlib` for paths; `tomllib` for TOML reads; `datetime.datetime.now(datetime.UTC)`
  not `utcnow()`.
- Iterate directly; `enumerate` when you need an index; `zip(..., strict=True)` when
  lengths must match.
- `match`/`case` when it is clearer than a ladder of `isinstance` / dict dispatch.
- Catch specific exceptions. No bare `except:` or `except Exception:` unless you log
  and re-raise. Use `logger.exception(...)` when logging a failure.
- Prefer `logging` over `print` in library and pipeline code.
- f-strings for formatting. No `%` or `.format` in new code unless required by an API.
- `secrets` for tokens; `random` only for non-security. Hash passwords with
  `hashlib.scrypt` / argon2 / bcrypt, never MD5/SHA1.

## Security

- No secrets, tokens, or private URLs in source. Read from the environment or a
  secret store.
- Parameterized SQL only. No f-strings / `%` / `.format` into queries.
- `subprocess.run([...], check=True)` with a list of args. No `shell=True`, no
  `os.system`.
- Do not `pickle.load` / `yaml.load` / `eval` / `exec` on untrusted data. Use
  `yaml.safe_load`.
- New dependencies: add via `uv add` (runtime) or `uv add --dev` (tools). Prefer
  stdlib. Justify anything heavy.

## Tests and layout

- Application code under `src/<package>/`; tests under top-level `tests/`.
- `test_*.py` / `test_*` functions. Independent tests; no shared mutable state.
- Assert exact values (`== expected`), not merely `is not None`. Use
  `pytest.raises(Error, match=...)` for error paths.
- Fixtures: `function` scope by default; `session` only for expensive immutable setup.
- Run with `uv run pytest`. Do not invoke `python setup.py test`.

## Tooling defaults (new or unconfigured projects)

Use **uv** + **Ruff** + **pytest**. Type-check with whatever the repo already runs
(Pyright, mypy, or ty); if none, prefer Pyright.

```bash
uv add --dev ruff pytest
uv run ruff check . --fix
uv run ruff format .
uv run pytest
```

Do not introduce Poetry, Pipenv, Black, isort, Flake8, or `requirements.txt` as the
source of truth when uv is available. Commit `uv.lock`. Install with
`uv sync --locked` in CI.

Ruff should own format + lint (`I`, `E`, `F`, `UP`, `B`, `SIM`). Do not bikeshed
quotes or import order in review; run the formatter.

## Review and write workflow

1. Read nearby files for local conventions (`line-length`, quotes, test style).
2. Choose the data-model tool with the decision list above.
3. Write typed, spaced, documented code. Public surface gets docstrings.
4. Handle errors and resources explicitly. No secrets, no injection.
5. Add or update tests for behavior you changed.
6. Before finishing, scan the diff against this checklist:

- [ ] 3.12+ types (`list[T]`, `X | None`, PEP 695 if generic)
- [ ] Readable spacing; no agglomerated blocks
- [ ] Correct model: Pydantic / dataclass / TypedDict / NamedTuple / Protocol
- [ ] No mutable defaults
- [ ] Google docstrings on the public API
- [ ] Context managers, specific exceptions
- [ ] Tests cover the change
- [ ] Ruff would pass (imports sorted, unused names gone)
