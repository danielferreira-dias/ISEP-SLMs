# SKINCON

## Local status

`data/annotations/` contains the complete public annotation files:

- `annotations_fitzpatrick17k.csv`: 3,690 rows; 3,230 are marked usable and
  460 have `Do not consider this image = 1`.
- `annotations_ddi.csv`: 656 rows; 636 are marked usable and 20 have
  `Do not consider this image = 1`.

Images are not redistributed by SKINCON. Obtain the corresponding
Fitzpatrick17k and DDI images under their respective access terms.

Source: <https://skincon-dataset.github.io/>.

## Dataset and labels

SKINCON is a multi-label morphology annotation layer. Each `ImageID` has 48
binary clinical-concept columns plus `Do not consider this image`.

The 48 concepts are:

```text
Vesicle, Papule, Macule, Plaque, Abscess, Pustule, Bulla, Patch,
Nodule, Ulcer, Crust, Erosion, Excoriation, Atrophy, Exudate,
Purpura/Petechiae, Fissure, Induration, Xerosis, Telangiectasia,
Scale, Scar, Friable, Sclerosis, Pedunculated, Exophytic/Fungating,
Warty/Papillomatous, Dome-shaped, Flat topped,
Brown(Hyperpigmentation), Translucent, White(Hypopigmentation),
Purple, Yellow, Black, Erythema, Comedo, Lichenification, Blue,
Umbilicated, Poikiloderma, Salmon, Wheal, Acuminate, Burrow, Gray,
Pigmented, Cyst
```

For example, one image can simultaneously have `Plaque = 1`, `Scale = 1`,
and `Erythema = 1`. These are visual findings, not mutually exclusive disease
classes.

In the downloaded Fitzpatrick17k annotation file the most frequent positive
concepts include erythema, plaque, papule, hyperpigmentation, and scale.

## Use and licensing

SKINCON is useful for morphology grounding, concept-bottleneck models, and
explainable image classification. Filter `Do not consider this image = 1`
before normal training.

The upstream image licences continue to apply. In particular, DDI's
individual agreement applies to the DDI image subset even though its SKINCON
annotations are public.

