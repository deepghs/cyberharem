import glob
import json
import os.path

import pandas as pd


def find_tags_from_workdir(workdir: str) -> pd.DataFrame:
    json_files = glob.glob(os.path.join(workdir, 'rtags', '*.json'))
    data = []
    for file in json_files:
        with open(file, 'r') as f:
            data.append(json.load(f))

    df = pd.DataFrame(data)
    df = df.sort_values(['index'], ascending=True)
    return df
