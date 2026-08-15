"""Command-line entry point for the aggregate SkinCAP observation audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.train.e2.skincap.audit import (
    audit_skincap_observations,
    write_audit_report,
)

DEFAULT_OUTPUT = Path("configs/datasets/skincap/observation_transform_audit.json")


def main() -> None:
    """Run the non-materializing audit and persist only aggregate metrics."""

    parser = argparse.ArgumentParser(
        description="Audit SkinCAP observation extraction without saving captions."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = audit_skincap_observations()
    write_audit_report(report, arguments.output)
    print(
        f"accepted={report.accepted_observation_rows} "
        f"candidates={report.technical_candidate_rows} "
        f"output={arguments.output}"
    )


if __name__ == "__main__":
    main()
