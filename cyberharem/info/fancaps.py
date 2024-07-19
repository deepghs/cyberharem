import os

import numpy as np
import pandas as pd
from ditk import logging
from hfutils.cache import delete_detached_cache
from tqdm import tqdm

from .myanimelist import search_from_myanimelist
from ..utils import get_hf_fs, get_hf_client

fancaps_repo = 'deepghs/fancaps_full'


def get_fancaps_bangumis():
    from .subsplease import _get_image_url

    hf_fs = get_hf_fs()
    hf_client = get_hf_client()

    fs_path = [
        os.path.relpath(path, f'datasets/{fancaps_repo}') for path in
        hf_fs.glob(f'datasets/{fancaps_repo}/tables/table-*.csv')
    ]
    bangumi_x = {}
    fancaps_records = []
    for path in fs_path:
        for df_chunk in tqdm(pd.read_csv(hf_client.hf_hub_download(
                repo_id=fancaps_repo,
                repo_type='dataset',
                filename=path,
        ), chunksize=10000), desc=f'Read Chunk {path!r}'):
            df_chunk = df_chunk.replace(np.NaN, None)
            for item in df_chunk.to_dict('records'):
                if item['bangumi_id'] not in bangumi_x:
                    logging.info(f'Bangumi found #{item["bangumi_id"]}, name: {item["bangumi_name"]!r}')
                    bangumi_x[item['bangumi_id']] = item['bangumi_name']
                    myanime_item = search_from_myanimelist(item['bangumi_name'])
                    if not myanime_item:
                        continue

                    fancaps_records.append({
                        'id': myanime_item['mal_id'],
                        'url': myanime_item['url'],
                        'approved': myanime_item['approved'],
                        'title': myanime_item['title'],
                        'title_english': myanime_item['title_english'],
                        'title_japanese': myanime_item['title_japanese'],
                        'type': myanime_item['type'],
                        'source': myanime_item['source'],
                        'episodes': myanime_item['episodes'],
                        'status': myanime_item['status'],
                        'airing': myanime_item['airing'],
                        'duration': myanime_item['duration'],
                        'rating': myanime_item['rating'],
                        'score': myanime_item['score'],
                        'scored_by': myanime_item['scored_by'],
                        'rank': myanime_item['rank'],
                        'popularity': myanime_item['popularity'],
                        'members': myanime_item['members'],
                        'favorites': myanime_item['favorites'],
                        'background': myanime_item['background'],
                        'season': myanime_item['season'],
                        'year': myanime_item['year'],
                        'cover_image_url': _get_image_url(myanime_item['images']),

                        'fancaps_id': item['bangumi_id'],
                        'fancaps_name': item['bangumi_name'],
                        'fancaps_url': f'https://fancaps.net/anime/showimages.php?{item["bangumi_id"]}',
                    })

    df_records = pd.DataFrame(fancaps_records)
    df_records = df_records.sort_values(by=['id'], ascending=[False])
    print(df_records)


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    delete_detached_cache()
    get_fancaps_bangumis()
