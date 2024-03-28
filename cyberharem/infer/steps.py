import glob
import json
import os.path
from typing import List

import pandas as pd
from imgutils.sd import read_metadata


def _raw_find_steps_in_workdir(workdir: str) -> List[dict]:
    last_attempt_file = os.path.join(workdir, 'last_attempt.json')
    if os.path.exists(last_attempt_file):
        with open(last_attempt_file, 'r') as f:
            last_attempt_info = json.load(f)

        if last_attempt_info['info']['reason'] == 'step_too_low':
            last_attempt_dir = os.path.join(workdir, last_attempt_info['rel_workdir'])
            df_last = pd.read_csv(os.path.join(last_attempt_dir, 'eval', 'metrics.csv'))
            p_steps, p_epochs = df_last['step'].max(), df_last['epoch'].max()
            data = _raw_find_steps_in_workdir(last_attempt_dir)

        else:
            p_steps, p_epochs = 0, 0
            data = []

    else:
        p_steps, p_epochs = 0, 0
        data = []

    exist_steps = set()
    for file in glob.glob(os.path.join(workdir, 'kohya', '*.safetensors')):
        meta = read_metadata(file)
        step = int(meta['ss_steps'])
        if step not in exist_steps:
            exist_steps.add(step)
            data.append({
                'step': p_steps + step,
                'epoch': p_epochs + int(meta['ss_epoch']),
                'name': os.path.splitext(os.path.basename(file))[0],
                'filename': os.path.basename(file),
                'workdir': os.path.abspath(workdir),
                'file': os.path.abspath(file),
            })

    return data


def find_steps_in_workdir(workdir: str) -> pd.DataFrame:
    data = _raw_find_steps_in_workdir(workdir)
    df = pd.DataFrame(data)
    df = df.sort_values(['step'], ascending=[True])
    return df
