from datasets import load_dataset

dataset = load_dataset(
    "danielfdias98/ISEPDermData",
    split="train",
    token=True,
)

image = dataset[0]["image"]
print(type(image))  # PIL image

image.show()