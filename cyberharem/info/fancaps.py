import datetime
import os

import numpy as np
import pandas as pd
import requests
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.cache import delete_detached_cache
from hfutils.operate import upload_directory_as_directory
from huggingface_hub import hf_hub_url
from tqdm import tqdm

from .myanimelist import search_from_myanimelist, _name_safe
from ..utils import get_hf_fs, get_hf_client, download_file, number_to_tag

fancaps_repo = 'deepghs/fancaps_full'


def get_fancaps_bangumis(repository: str):
    from .subsplease import _get_image_url

    hf_fs = get_hf_fs()
    hf_client = get_hf_client()
    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset')

    fs_path = [
        os.path.relpath(path, f'datasets/{fancaps_repo}') for path in
        hf_fs.glob(f'datasets/{fancaps_repo}/tables/table-*.csv')
    ]
    bangumi_x = {}
    bangumi_eps = {}
    bangumi_images = {}
    fancaps_records = []
    for path in fs_path:
        for df_chunk in tqdm(pd.read_csv(hf_client.hf_hub_download(
                repo_id=fancaps_repo,
                repo_type='dataset',
                filename=path,
        ), chunksize=10000), desc=f'Read Chunk {path!r}'):
            df_chunk = df_chunk.replace(np.NaN, None)
            for item in df_chunk.to_dict('records'):
                if item['bangumi_id'] not in bangumi_eps:
                    bangumi_eps[item['bangumi_id']] = set()
                bangumi_eps[item['bangumi_id']].add(item['episode_id'])
                bangumi_images[item['bangumi_id']] = bangumi_images.get(item['bangumi_id'], 0) + 1

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

            if len(fancaps_records) > 10:
                break
        if len(fancaps_records) > 10:
            break

    df_animes = pd.DataFrame(fancaps_records)
    df_animes['fancaps_episodes'] = [len(bangumi_eps[x]) for x in df_animes['fancaps_id']]
    df_animes['fancaps_images'] = [bangumi_images[x] for x in df_animes['fancaps_id']]
    df_animes = df_animes.sort_values(by=['id'], ascending=[False])
    df_animes = df_animes.replace(np.NaN, None)
    logging.info(f'Animes:\b{df_animes}')
    with TemporaryDirectory() as td:
        animes_file = os.path.join(td, 'animes.parquet')
        df_animes.to_parquet(animes_file, engine='pyarrow', index=False)

        images_dir = os.path.join(td, 'images')
        os.makedirs(images_dir, exist_ok=True)

        l_shown = []
        for aitem in tqdm(df_animes.to_dict('records'), desc='Dump Items'):
            logging.info(f'Making item for {aitem["id"]}, title: {aitem["title"]!r} ...')
            safe_bangumi_name = aitem['title'].replace('`', ' ').replace('[', '(').replace(']', ')')

            suffix = f'{aitem["id"]}__{_name_safe(aitem["title"]).replace(" ", "_").lower()}'
            post_file = os.path.join(images_dir, f'{suffix}.jpg')
            os.makedirs(os.path.dirname(post_file), exist_ok=True)
            if aitem['cover_image_url']:
                try:
                    download_file(aitem['cover_image_url'], post_file)
                except requests.exceptions.HTTPError as err:
                    if err.response.status_code == 404:
                        logging.warning(f'Post file 404 for bangumi {aitem["title"]!r} ...')
                        post_file = None
                    else:
                        raise err
            else:
                post_file = None
            post_url = hf_hub_url(repo_id=repository, repo_type='dataset', filename=os.path.relpath(post_file, td))
            post_md = f'![{suffix}]({post_url})' if post_file else '(no post)'
            if aitem['url']:
                post_md = f'[{post_md}]({aitem["url"]})'

            l_shown.append({
                'ID': aitem['id'],
                'Post': post_md,
                'Bangumi': f'[{safe_bangumi_name}]({aitem["fancaps_url"]})',
                'Type': aitem['type'],
                'Episodes': f'{aitem["fancaps_episodes"]} / {int(aitem["episodes"]) if aitem["episodes"] else "?"}',
                'Images': aitem['fancaps_images'],
                'Status': aitem['status'] if aitem['airing'] else f'**{aitem["status"]}**',
                'Score': aitem['score'],
            })
        df_l_shown = pd.DataFrame(l_shown)

        with open(os.path.join(td, 'README.md'), 'w') as f:
            print('---', file=f)
            print('license: other', file=f)
            print('language:', file=f)
            print('- en', file=f)
            print('tags:', file=f)
            print('- anime', file=f)
            print('size_categories:', file=f)
            print(f'- {number_to_tag(len(df_animes))}', file=f)
            print('source_datasets:', file=f)
            print('- myanimelist', file=f)
            print('- fancaps', file=f)
            print('---', file=f)
            print('', file=f)

            print('This is an integration database of fancaps, myanimelist. '
                  'You can know which animes have been scrapped by fancaps already.', file=f)
            print('', file=f)
            print('This database is refreshed daily.', file=f)
            print('', file=f)

            print(f'## Current Animes', file=f)
            print('', file=f)
            current_time = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
            print(f'{plural_word(len(df_animes), "anime")} in total, '
                  f'Last updated on: `{current_time}`.', file=f)
            print('', file=f)
            print(df_l_shown.to_markdown(index=False), file=f)
            print('', file=f)

        os.system(f'tree {td!r}')

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=td,
            path_in_repo='.',
            message=f'Sync {plural_word(len(df_animes), "anime")} from fancaps',
            clear=True,
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    delete_detached_cache()
    get_fancaps_bangumis(
        repository='deepghs/fancaps_animes',
    )
