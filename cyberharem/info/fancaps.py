import os

import pandas as pd

from ..utils import get_hf_fs, get_hf_client

fancaps_repo = 'deepghs/fancaps_full'


def get_fancaps_bangumis():
    hf_fs = get_hf_fs()
    hf_client = get_hf_client()

    fs_path = [
        os.path.relpath(path, f'datasets/{fancaps_repo}') for path in
        hf_fs.glob(f'datasets/{fancaps_repo}/tables/table-*.csv')
    ]
    for path in fs_path:
        for df_chunk in pd.read_csv(hf_client.hf_hub_download(
                repo_id=fancaps_repo,
                repo_type='dataset',
                filename=path,
        ), chunksize=10000):
            print(df_chunk)


if __name__ == '__main__':
    get_fancaps_bangumis()
