import glob
import os.path

import pandas as pd
from imgutils.sd import read_metadata


def find_steps_in_workdir(workdir: str) -> pd.DataFrame:
    data = []
    exist_steps = set()
    for file in glob.glob(os.path.join(workdir, 'kohya', '*.safetensors')):
        meta = read_metadata(file)
        step = int(meta['ss_steps'])
        if step not in exist_steps:
            exist_steps.add(step)
            data.append({
                'step': step,
                'epoch': int(meta['ss_epoch']),
                'name': os.path.splitext(os.path.basename(file))[0],
                'filename': os.path.basename(file),
                'file': os.path.abspath(file),
            })
    df = pd.DataFrame(data)
    df = df.sort_values(['step'], ascending=[True])
    return df
