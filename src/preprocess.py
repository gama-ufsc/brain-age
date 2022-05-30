from os import remove
from pathlib import Path
from shutil import move

import pandas as pd

from tqdm import tqdm

from brats.preprocessing.nipype_wrappers import ants_registration, ants_transformation


INPUT_ADNI_DIR = Path('/data/slow/ADNI')
OUTPUT_ADNI_DIR = Path('/data/slow/ADNI_prep')

TEMPLATE_FPATH = Path('/home/bruno-pacheco/brain-age/data/external/SRI24_T1.nii')

tmpdir = Path('.tmpdir')

if __name__ == '__main__':
    assert INPUT_ADNI_DIR.exists(), f"`{INPUT_ADNI_DIR}` doesn't exist"
    assert TEMPLATE_FPATH.exists(), f"`{TEMPLATE_FPATH}` doesn't exist"

    OUTPUT_ADNI_DIR.mkdir(exist_ok=True)

    tmpdir.mkdir(exist_ok=True)

    metadata_fpath = next(INPUT_ADNI_DIR.glob('*.csv'))
    df = pd.read_csv(metadata_fpath)

    for img_fpath in tqdm(list(INPUT_ADNI_DIR.glob('**/*.nii'))):
        local_fpath = str(img_fpath).lstrip(str(INPUT_ADNI_DIR))
        subject_id = local_fpath.split('/')[0]

        image_id = img_fpath.name.split('_')[-1][:-4]

        img_meta = df[(df['Subject'] == subject_id) & (df['Image Data ID'] == image_id)]
        assert img_meta.shape[0] == 1, str(img_meta)

        subject_age = img_meta.iloc[0]['Age']

        output_fpath = OUTPUT_ADNI_DIR/f"{subject_id}__{image_id}__{subject_age}.nii"

        if output_fpath.exists():
            continue

        reg_transform, _ = ants_registration(
            str(TEMPLATE_FPATH),
            str(img_fpath),
            str(tmpdir/'transf_'),
        )

        prep_fpath = ants_transformation(
            str(img_fpath),
            str(TEMPLATE_FPATH),
            [reg_transform,],
            str(tmpdir/'sri24_'),
        )

        move(prep_fpath, output_fpath)

        try:
            remove(reg_transform)
        except FileNotFoundError:
            pass
