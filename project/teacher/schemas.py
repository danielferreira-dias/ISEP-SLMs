"""Pydantic models for teacher JSON: LLM output, manifest rows, and JSONL records."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RecordStatus(StrEnum):
    """JSONL status written by Stage A and Stage B."""

    OK = "ok"
    ERROR = "error"
    REJECTED = "rejected"


class CiteField(StrEnum):
    """Stage A fields that Stage B may cite as evidence."""

    PRIMARY_LESION = "primary_lesion"
    SIZE = "size"
    COLOR = "color"
    SHAPE = "shape"
    BORDER = "border"
    SURFACE = "surface"
    SECONDARY_MORPHOLOGY = "secondary_morphology"
    CONFIGURATION = "configuration"
    DISTRIBUTION = "distribution"
    ADDITIONAL_FEATURES = "additional_features"


class ImageModality(StrEnum):
    """Acquisition type declared by Stage A."""

    CLINICAL = "clinical"
    DERMOSCOPY = "dermoscopy"
    UNKNOWN = "unknown"


DERMOSCOPIC_ONLY_FEATURES: frozenset[str] = frozenset(
    {
        "blue-white veil",
        "pigment network",
        "atypical pigment network",
        "dots and globules",
        "streaks",
        "regression structures",
        "structureless area",
        "shiny white lines",
        "milia-like cysts",
    }
)

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid")


class StageAMorphology(BaseModel):
    """Validated Stage A teacher JSON. Visual findings only."""

    model_config = _MODEL_CONFIG

    image_quality: Literal["evaluable", "limited", "not_evaluable"]
    modality: ImageModality
    primary_lesion: Literal[
        "macule",
        "patch",
        "papule",
        "plaque",
        "nodule",
        "tumor",
        "vesicle",
        "bulla",
        "pustule",
        "wheal",
        "cyst",
        "comedo",
        "burrow",
        "ulcer",
        "erosion",
        "not_assessable",
    ]
    size: Literal[
        "cannot_assess",
        "punctate",
        "few_mm",
        "about_1_cm",
        "few_cm",
        "large_or_extensive",
    ]
    color: list[
        Literal[
            "brown",
            "black",
            "red",
            "pink",
            "purple",
            "blue",
            "gray",
            "white",
            "yellow",
            "orange",
            "skin-colored",
            "hypopigmented",
            "not_assessable",
        ]
    ]
    shape: Literal[
        "round",
        "oval",
        "irregular",
        "symmetric",
        "asymmetric",
        "not_assessable",
    ]
    border: Literal[
        "well-defined",
        "ill-defined",
        "regular",
        "irregular",
        "raised",
        "fading",
        "not_assessable",
    ]
    surface: Literal[
        "flat",
        "elevated",
        "smooth",
        "keratotic",
        "verrucous",
        "exudative",
        "not_assessable",
    ]
    secondary_morphology: list[
        Literal[
            "scale",
            "crust",
            "erosion",
            "ulceration",
            "fissure",
            "lichenification",
            "excoriation",
            "atrophy",
        ]
    ]
    configuration: Literal[
        "solitary",
        "grouped",
        "clustered",
        "annular",
        "arcuate",
        "linear",
        "serpiginous",
        "polycyclic",
        "targetoid",
        "scattered",
        "disseminated",
        "not_assessable",
    ]
    distribution: Literal[
        "localized",
        "generalized",
        "acral",
        "flexural",
        "extensor",
        "photodistributed",
        "intertriginous",
        "dermatomal",
        "not_assessable",
    ]
    additional_features: list[
        Literal[
            "heterogeneous pigmentation",
            "homogeneous pigmentation",
            "blue-white veil",
            "pigment network",
            "atypical pigment network",
            "dots and globules",
            "streaks",
            "regression structures",
            "structureless area",
            "shiny white lines",
            "telangiectasia",
            "milia-like cysts",
            "follicular plugging",
            "central scar",
            "satellite lesions",
            "collarette scale",
            "purpura",
            "necrosis",
            "weeping",
        ]
    ]


class EvidenceCitation(BaseModel):
    """One Stage A field/value pair cited by a differential item."""

    model_config = _MODEL_CONFIG

    field: CiteField
    value: str = Field(min_length=1)


class DifferentialItem(BaseModel):
    """One ranked hypothesis in the Stage B differential."""

    model_config = _MODEL_CONFIG

    rank: int = Field(ge=1, le=5)
    disease: str = Field(min_length=1)
    supporting: list[EvidenceCitation]
    contradicting: list[EvidenceCitation]
    missing: list[
        Literal[
            "duration_and_evolution",
            "symptoms",
            "palpation",
            "other_body_sites",
            "dermoscopy",
            "scale_or_ruler",
            "closer_or_overview_image",
            "clinical_history",
        ]
    ]


class StageBReasoning(BaseModel):
    """Validated Stage B teacher JSON. Gold-anchored differential."""

    model_config = _MODEL_CONFIG

    differential_diagnosis: list[DifferentialItem] = Field(min_length=2, max_length=5)
    reasoning: str = Field(min_length=1)
    diagnosis: str = Field(min_length=1)


class ManifestRow(BaseModel):
    """One line of the generation manifest."""

    model_config = _MODEL_CONFIG

    sample_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    gold_diagnosis: str = Field(min_length=1)

    @field_validator("sample_id", "gold_diagnosis")
    @classmethod
    def _strip_nonempty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class ImageSample(BaseModel):
    """Stage A input: identity and image only. No gold diagnosis."""

    model_config = _MODEL_CONFIG

    sample_id: str = Field(min_length=1)
    image_path: Path


class UsageInfo(BaseModel):
    """Token and cost fields copied from an OpenRouter response."""

    model_config = _MODEL_CONFIG

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


class StageAFileRow(BaseModel):
    """One Stage A JSONL record. Must not contain gold_diagnosis."""

    model_config = _MODEL_CONFIG

    sample_id: str
    status: RecordStatus
    morphology: StageAMorphology | None = None
    error: str | None = None
    usage: UsageInfo | None = None
    teacher: str
    image_path: str


class StageBFileRow(BaseModel):
    """One Stage B JSONL record. Rejected rows stay auditable."""

    model_config = _MODEL_CONFIG

    sample_id: str
    status: RecordStatus
    reasoning: StageBReasoning | None = None
    reasons: tuple[str, ...] = ()
    error: str | None = None
    usage: UsageInfo | None = None
    teacher: str
    gold_diagnosis: str
    stage_a_sample_id: str
    image_path: str


def parse_stage_a(raw: dict[str, object]) -> StageAMorphology:
    """Validate untrusted Stage A JSON from the teacher.

    Args:
        raw: Parsed object from the model content string.

    Returns:
        A frozen morphology record.

    Raises:
        ValidationError: If a field is missing, extra, or off-enum.
    """
    return StageAMorphology.model_validate(raw)


def parse_stage_b(raw: dict[str, object]) -> StageBReasoning:
    """Validate untrusted Stage B JSON from the teacher.

    Args:
        raw: Parsed object from the model content string.

    Returns:
        A frozen reasoning record.

    Raises:
        ValidationError: If the differential or diagnosis is invalid.
    """
    return StageBReasoning.model_validate(raw)


def image_sample_from_manifest(row: ManifestRow, *, project_root: Path) -> ImageSample:
    """Drop gold and resolve the image path against ``project/``.

    Args:
        row: Manifest line that still carries gold_diagnosis.
        project_root: Directory used for relative image paths.

    Returns:
        Stage A input with an absolute or project-relative Path.
    """
    path = Path(row.image_path)
    if not path.is_absolute():
        path = project_root / path

    return ImageSample(sample_id=row.sample_id, image_path=path)
