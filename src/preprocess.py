import pickle
import re
import xml.etree.ElementTree as ET

from os import remove
from pathlib import Path
from shutil import move

import pandas as pd

from tqdm import tqdm

from brats.preprocessing.nipype_wrappers import ants_registration, ants_transformation


INPUT_ADNI_DIR = Path('/home/jupyter/gama/bruno/data/raw/ADNI1')
OUTPUT_ADNI_DIR = Path('/home/jupyter/gama/bruno/data/raw/ADNI1_prep')

TEMPLATE_FPATH = Path('/home/jupyter/gama/bruno/data/external/SRI24_T1.nii')

tmpdir = Path('.tmpdir')

if __name__ == '__main__':
    assert INPUT_ADNI_DIR.exists(), f"`{INPUT_ADNI_DIR}` doesn't exist"
    assert TEMPLATE_FPATH.exists(), f"`{TEMPLATE_FPATH}` doesn't exist"

    OUTPUT_ADNI_DIR.mkdir(exist_ok=True)

    tmpdir.mkdir(exist_ok=True)

    ages = dict()

    for img_fpath in tqdm(list(INPUT_ADNI_DIR.glob('**/*.nii'))):
        local_fpath = str(img_fpath).lstrip(str(INPUT_ADNI_DIR))
        subject_id = local_fpath.split('/')[0]

        image_id = img_fpath.name.split('_')[-1][:-4]

        # get age from metadata
        meta_fpath = INPUT_ADNI_DIR/re.sub(r"(?P<scaled>_Scaled)(?P<two>_2|)(?P<del>_Br_[0-9]*)(?P<end>_S)", '\g<scaled>\g<two>\g<end>', img_fpath.name).replace('.nii', '.xml').replace('_MR_MPR', '_MPR')
        meta = ET.parse(meta_fpath)
        ages[image_id] = float(meta.find('project').find('subject').find('study').find('subjectAge').text)

        output_fpath = OUTPUT_ADNI_DIR/f"{subject_id}__{image_id}.nii"

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

    with open(OUTPUT_ADNI_DIR/'ages.pkl', 'wb') as f:
        pickle.dump(ages)
