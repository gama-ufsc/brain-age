import pickle
import re
import xml.etree.ElementTree as ET

from os import remove
from pathlib import Path
from shutil import move

import nibabel as nib
import pandas as pd

from tqdm import tqdm

from brats.preprocessing.nipype_wrappers import ants_registration, ants_transformation


INPUT_ADNI_DIR = Path('/home/jupyter/gama/bruno/data/raw/ADNI1')
OUTPUT_ADNI_DIR = Path('/home/jupyter/gama/bruno/data/interim/ADNI1_4bashyam')

TEMPLATE_FPATH = Path('/home/jupyter/gama/bruno/data/external/MNI152_T1_1mm_brain_LPS_filled.nii.gz')

tmpdir = Path('.tmpdir')

if __name__ == '__main__':
    assert INPUT_ADNI_DIR.exists(), f"`{INPUT_ADNI_DIR}` doesn't exist"
    assert TEMPLATE_FPATH.exists(), f"`{TEMPLATE_FPATH}` doesn't exist"

    OUTPUT_ADNI_DIR.mkdir(exist_ok=True)

    tmpdir.mkdir(exist_ok=True)

    ages = dict()
    groups = dict()

    for img_fpath in tqdm(list(INPUT_ADNI_DIR.glob('**/*.nii'))):
        local_fpath = str(img_fpath).lstrip(str(INPUT_ADNI_DIR))
        subject_id = local_fpath.split('/')[0]

        match = re.search(r"(?P<image_id>I[0-9]+)\.nii", img_fpath.name)
        image_id = match['image_id']

        # get age from metadata
        meta_fpath = next(Path(INPUT_ADNI_DIR).glob(f"*{image_id}.xml"))
        meta = ET.parse(meta_fpath)
        ages[image_id] = float(meta.find('project').find('subject').find('study').find('subjectAge').text)
        groups[image_id] = meta.find('project').find('subject').find('researchGroup').text

        output_fpath = OUTPUT_ADNI_DIR/f"{subject_id}__{image_id}.nii.gz"

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

#         move(prep_fpath, output_fpath)
        nib.save(nib.load(prep_fpath), output_fpath)

        try:
            remove(reg_transform)
            remove(prep_fpath)
        except FileNotFoundError:
            pass

    with open(OUTPUT_ADNI_DIR/'groups.pkl', 'wb') as f:
        pickle.dump(groups, f)

    with open(OUTPUT_ADNI_DIR/'ages.pkl', 'wb') as f:
        pickle.dump(ages, f)
