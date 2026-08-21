"""Turn loaded ISEPDistillDataset rows into generation examples."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from PIL import Image

from project.dataset.dataset import DistillDataset


@dataclass(slots=True, kw_only=True, frozen=True)
class DistillExample:
    """One Hub row ready for Stage A/B. Stage A must ignore gold_diagnosis."""

    sample_id: str
    gold_diagnosis: str
    image: Image.Image
    source_ref: str


def iter_distill_examples(
    *,
    config: str = "diagnosis",
    split: str = "sft_train",
) -> Iterator[DistillExample]:
    """Yield examples from one Hub config/split.

    Args:
        config: Dataset config name, usually ``diagnosis``.
        split: Split to generate from. Default ``sft_train``.

    Yields:
        Independent PIL images with sample_id and gold_diagnosis.

    Raises:
        TypeError: If a row is missing image, sample_id, or gold_diagnosis.
    """
    dataset = DistillDataset.load(config=config, split=split)
    table = cast(Iterable[object], dataset.get(config, split))

    for row in table:
        if not isinstance(row, Mapping):
            raise TypeError("Hub row is not a mapping")
        yield example_from_hub_row(
            row,
            repo_id=dataset.spec.huggingface.repo_id,
            revision=dataset.spec.huggingface.revision,
            config=config,
            split=split,
        )


def example_from_hub_row(
    row: Mapping[str, object],
    *,
    repo_id: str,
    revision: str,
    config: str,
    split: str,
) -> DistillExample:
    """Build one DistillExample from a Hub mapping row.

    Args:
        row: Dataset row with image, sample_id, and gold_diagnosis.
        repo_id: Immutable Hub repository identity.
        revision: Immutable Hub commit used for this generation.
        config: Hub config name, used only in source_ref.
        split: Hub split name, used only in source_ref.

    Returns:
        A generation example. The image is copied without discarding EXIF/ICC;
        deterministic preprocessing occurs immediately before the API call.

    Raises:
        TypeError: If required fields are missing or the wrong type.
    """
    sample_id = row.get("sample_id")
    gold = row.get("gold_diagnosis")
    image = row.get("image")

    if not isinstance(sample_id, str) or not sample_id.strip():
        raise TypeError("Hub row is missing sample_id")
    if not isinstance(gold, str) or not gold.strip():
        raise TypeError(f"Hub row {sample_id!r} is missing gold_diagnosis")
    if not isinstance(image, Image.Image):
        raise TypeError(f"Hub row {sample_id!r} image is {type(image).__name__}")

    copied = image.copy()
    return DistillExample(
        sample_id=sample_id.strip(),
        gold_diagnosis=gold.strip(),
        image=copied,
        source_ref=(
            f"hf://datasets/{repo_id}@{revision}/{config}/{split}/{sample_id.strip()}"
        ),
    )


def examples_from_manifest(path: Path, *, project_root: Path) -> list[DistillExample]:
    """Load a file-path manifest into DistillExample rows.

    Args:
        path: JSONL with sample_id, image_path, gold_diagnosis.
        project_root: Root for relative image paths.

    Returns:
        Examples with independent image copies from each file.
    """
    from project.teacher.utils.jsonl import load_manifest

    examples: list[DistillExample] = []
    for row in load_manifest(path):
        image_path = Path(row.image_path)
        if not image_path.is_absolute():
            image_path = project_root / image_path
        with Image.open(image_path) as image:
            examples.append(
                DistillExample(
                    sample_id=row.sample_id,
                    gold_diagnosis=row.gold_diagnosis,
                    image=image.copy(),
                    source_ref=str(image_path),
                )
            )
    return examples
