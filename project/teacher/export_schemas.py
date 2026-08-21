"""Export canonical Pydantic teacher schemas to the checked-in JSON snapshots."""

import json
from pathlib import Path

from project.teacher.schemas import teacher_output_schema

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_DESTINATIONS = {
    "A": PROJECT_ROOT / "configs" / "schemas" / "stage_a_morphology.json",
    "B": PROJECT_ROOT / "configs" / "schemas" / "stage_b_reasoning.json",
}


def export_teacher_schemas() -> tuple[Path, ...]:
    """Write deterministic JSON snapshots and return their paths."""
    written: list[Path] = []
    for stage_key, path in _DESTINATIONS.items():
        payload = teacher_output_schema(stage_key)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return tuple(written)


def main() -> None:
    """Export both schemas for maintainers and CI."""
    for path in export_teacher_schemas():
        print(path)


if __name__ == "__main__":
    main()
