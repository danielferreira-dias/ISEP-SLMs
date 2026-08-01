# HIBA Skin Lesions

## Purpose in this repository

HIBA is a **training-only** source. The official release contains 1,635
images, but this project keeps only its 355 clinical photographs:

| Source image type | Images | Used for training |
| --- | ---: | --- |
| Clinical overview | 349 | Yes |
| Clinical close-up | 6 | Yes |
| Dermoscopic | 1,280 | No |

The clinical subset represents 354 lesions from 248 patients. Rows are grouped
by `patient_id` so related lesions cannot later be treated as independent
patients.

## Labels and metadata

The main label is `diagnosis`. Clinical-image examples include basal cell
carcinoma, melanoma, nevus, squamous cell carcinoma, dermatofibroma,
seborrheic keratosis, actinic keratosis, vascular lesion, solar lentigo, and
lichenoid keratosis. Diagnoses outside the thesis's 21-class taxonomy remain
useful as `training_role=out_of_domain`; they do not silently become a
closed-set class.

Other fields include:

- `lesion_id` and `patient_id`
- `diagnosis_confirm_type`
- anatomical site
- age and sex
- Fitzpatrick skin type
- benign/malignant status

## Storage and licence

The CSV and official ZIP archive live under `data/`; the images remain
compressed and are addressed through `zip://...::member` locators. Local
payloads are ignored by Git.

HIBA is CC BY 4.0.

Source: <https://api.isic-archive.com/doi/hiba-skin-lesions/>
