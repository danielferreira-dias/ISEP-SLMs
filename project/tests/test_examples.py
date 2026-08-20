"""Hub row → DistillExample conversion."""

from PIL import Image

from project.dataset.examples import example_from_hub_row


def test_example_from_hub_row_copies_rgb_and_gold() -> None:
    image = Image.new("RGB", (12, 8), "green")
    example = example_from_hub_row(
        {
            "sample_id": "abc",
            "gold_diagnosis": "melanoma",
            "image": image,
        },
        config="diagnosis",
        split="sft_train",
    )
    assert example.sample_id == "abc"
    assert example.gold_diagnosis == "melanoma"
    assert example.source_ref == "hf://diagnosis/sft_train/abc"
    assert example.image.mode == "RGB"
    assert example.image.size == (12, 8)
    assert example.image is not image
