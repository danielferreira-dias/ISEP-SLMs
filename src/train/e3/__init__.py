"""E3 hard-KD contracts and deterministic teacher-target rendering."""

from src.train.e3.domain import (
    ClinicalAction,
    ConfidenceLevel,
    DifferentialDiagnosis,
    ImageAssessment,
    MissingDiscriminator,
    Observation,
    ObservationStatus,
    RiskLevel,
    StructuredClinicalTarget,
)
from src.train.e3.phase import (
    E3FormattedExample,
    E3StructuredPhase,
    E3StructuredSample,
    E3TrainingVariant,
)
from src.train.e3.rendering import (
    DeterministicOpenResponseRenderer,
    RenderedOpenResponse,
    canonical_structured_json,
)
from src.train.e3.templates import OPEN_RESPONSE_TEMPLATES, OpenResponseTemplate

__all__ = [
    "OPEN_RESPONSE_TEMPLATES",
    "ClinicalAction",
    "ConfidenceLevel",
    "DeterministicOpenResponseRenderer",
    "DifferentialDiagnosis",
    "E3FormattedExample",
    "E3StructuredPhase",
    "E3StructuredSample",
    "E3TrainingVariant",
    "ImageAssessment",
    "MissingDiscriminator",
    "Observation",
    "ObservationStatus",
    "OpenResponseTemplate",
    "RenderedOpenResponse",
    "RiskLevel",
    "StructuredClinicalTarget",
    "canonical_structured_json",
]
