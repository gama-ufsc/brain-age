# Pre-training brain age deep learning models

This repository contains the development for the paper "Does pre-training on brain-related tasks results in better deep-learning-based brain age biomarkers?", submitted to BRACIS'23 (citation soon).

We use wandb to keep models and results.
The `train_brainage.py` and `train_brats_models.py` are the basis for reproducing our results.
Of course, they assume you already have the data ready for training.

The ADNI data we use are the standard split for ADNI1 (which is already a well-defined collection in loni's platform) and the images specified in `data/raw/ADNI*_image_ids.csv`.
Scripts `src/preprocess.py` and `src/make_dataset.py` do the job of getting the data ready for training.
With respect to BraTS pre-training, you must follow the steps of [nnUNet](https://github.com/MIC-DKFZ/nnUNet) to prepare the data.

The brain age prediction results can be seen with detail in our [W&B report](https://api.wandb.ai/links/gama/27wjeec2).
All statistical analysis mentioned in the paper come from `statistical_analysis_results.csv`.
