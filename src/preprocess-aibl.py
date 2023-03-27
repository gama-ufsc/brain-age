from os import remove
from pathlib import Path
from shutil import move

import pandas as pd

from tqdm import tqdm
import sys
parentdir = '/home/jupyter/gama/brats'
sys.path.insert(0, parentdir)

from brats.preprocessing.nipype_wrappers import ants_registration, ants_transformation

INPUT_ADNI_DIR = Path('/home/jupyter/data/AIBL/AIBL_dissertacao')
OUTPUT_ADNI_DIR = Path('/home/jupyter/data/AIBL/AIBL_dissertacao_prep')

TEMPLATE_FPATH = Path('/home/jupyter/gama/brain-age/data/SRI24_T1.nii')

tmpdir = Path('.tmpdir')

if __name__ == '__main__':
    assert INPUT_ADNI_DIR.exists(), f"`{INPUT_ADNI_DIR}` doesn't exist"
    assert TEMPLATE_FPATH.exists(), f"`{TEMPLATE_FPATH}` doesn't exist"

    OUTPUT_ADNI_DIR.mkdir(exist_ok=True)

    tmpdir.mkdir(exist_ok=True)

    df = pd.read_csv('/home/jupyter/data/AIBL/AIBL_dissertacao/AIBL_dissertacao_data.csv')
    for img_fpath in tqdm(list(INPUT_ADNI_DIR.glob('**/*.nii.gz'))):
        img_fpath = str(img_fpath)
        subject_id = img_fpath.split('/')[-1].split('.nii')[0]

        img_meta = df.query('imageid == @subject_id')


        subject_age = img_meta.iloc[0]['age']

        output_fpath = OUTPUT_ADNI_DIR/f"{subject_id}__{subject_age}.nii.gz"

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
