import datetime
import fnmatch
import json
import logging
import os.path
import textwrap
import time
import warnings
from typing import Tuple, Optional

import dateparser
import pandas as pd
import requests.exceptions
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from huggingface_hub import CommitOperationAdd
from pyanimeinfo.myanimelist import JikanV4Client
from requests.exceptions import HTTPError
from tqdm.auto import tqdm

from ...utils import get_hf_client, get_hf_fs, download_file, get_global_bg_namespace

hf_client = get_hf_client()
hf_fs = get_hf_fs()

client = JikanV4Client()


def _get_url_from_small_dict(dict_: dict):
    if 'maximum_image_url' in dict_:
        return dict_['maximum_image_url']
    elif 'large_image_url' in dict_:
        return dict_['large_image_url']
    elif 'small_image_url' in dict_:
        return dict_['small_image_url']
    else:
        return dict_['image_url']


def _get_image_url(image_dict: dict):
    if 'jpg' in image_dict:
        return _get_url_from_small_dict(image_dict['jpg'])
    elif 'webp' in image_dict:
        return _get_url_from_small_dict(image_dict['webp'])
    else:
        return None


def get_animelist_info(bangumi_name, bangumi_id: Optional[int] = None) \
        -> Tuple[Optional[int], Optional[str], Optional[str]]:
    while True:
        try:
            items = client.search_anime(bangumi_name)
        except HTTPError as err:
            if err.response.status_code == 429:
                warnings.warn(f'429 error detected: {err!r}, wait for some seconds ...')
                time.sleep(5.0)
            else:
                raise
        else:
            break

    # noinspection DuplicatedCode
    if not items:
        return None, None, None

    all_items = []
    for i, item_ in enumerate(items):
        if (bangumi_id and item_['mal_id'] == bangumi_id) or \
                (item_['type'] and item_['type'].lower() in {"tv", "movie", "ova", "special", "ona"}):
            all_items.append((
                0 if (bangumi_id and item_['mal_id'] == bangumi_id) else 1,
                i,
                item_
            ))

    if not all_items:
        return None, None, None
    _, _, item = sorted(all_items)[0]

    bangumi_url = item['url']
    image_url = _get_image_url(item['images'])

    return item['mal_id'], bangumi_url, image_url


def sync_bangumi_base(repository: str = f'{get_global_bg_namespace()}/README'):
    cb_models = [item.modelId for item in hf_client.list_models(author='CyberHarem')]
    cb_datasets = [item.id for item in hf_client.list_datasets(author='CyberHarem')]

    with TemporaryDirectory() as td:
        readme_file = os.path.join(td, 'README.md')
        with open(readme_file, 'w') as f:
            rows, total_images, total_clusters, total_animes = [], 0, 0, 0
            data_rows = []
            for item in tqdm(list(hf_client.list_datasets(author=get_global_bg_namespace()))):
                if item.private:
                    logging.info(f'Repo {item.id!r} is private, skipped.')
                    continue
                if not hf_client.file_exists(
                        repo_id=item.id,
                        repo_type='dataset',
                        filename='meta.json',
                ):
                    logging.info(f'No meta information found for {item.id!r}, skipped')
                    continue

                with open(hf_client.hf_hub_download(
                        repo_id=item.id,
                        repo_type='dataset',
                        filename='meta.json',
                ), 'r') as mf:
                    meta = json.load(mf)
                bangumi_name = meta['name']
                safe_bangumi_name = bangumi_name.replace('`', ' ').replace('[', '(').replace(']', ')')
                suffix = item.id.split('/')[-1]
                datasets_cnt = len([x for x in cb_datasets if fnmatch.fnmatch(x, f'CyberHarem/*_{suffix}')])
                models_cnt = len([x for x in cb_models if fnmatch.fnmatch(x, f'CyberHarem/*_{suffix}')])

                logging.info(f'Getting post url for {bangumi_name!r} ...')
                anime_id, page_url, post_url = get_animelist_info(
                    bangumi_name, bangumi_id=meta.get('myanimelist_id'))
                if post_url:
                    post_file = os.path.join(td, 'posts', f'{suffix}.jpg')
                    os.makedirs(os.path.dirname(post_file), exist_ok=True)
                    try:
                        download_file(post_url, post_file)
                    except requests.exceptions.HTTPError as err:
                        if err.response.status_code == 404:
                            logging.warning(f'Post file 404 for bangumi {bangumi_name!r} ...')
                            post_file = None
                        else:
                            raise err
                else:
                    post_file = None

                dataset_url = f'https://huggingface.co/datasets/{item.id}'
                post_md = f'![{suffix}]({os.path.relpath(post_file, td)})' if post_file else '(no post)'
                if page_url:
                    post_md = f'[{post_md}]({page_url})'
                last_modified = dateparser.parse(item.lastModified) \
                    if isinstance(item.lastModified, str) else item.lastModified
                rows.append({
                    'Post': post_md,
                    'Bangumi': f'[{safe_bangumi_name}]({dataset_url})',
                    'Last Modified': last_modified.strftime('%Y-%m-%d %H:%M'),
                    'Images': meta['total'],
                    'Clusters': len([x for x in meta['ids'] if x != -1]),
                    'Datasets': f'[{datasets_cnt}](https://huggingface.co/CyberHarem?'
                                f'search_models=_{suffix}&search_datasets=_{suffix})',
                    'Models': f'[{models_cnt}](https://huggingface.co/CyberHarem?'
                              f'search_models=_{suffix}&search_datasets=_{suffix})',
                })
                data_rows.append({
                    'id': anime_id or -1,
                    'myanimelist_url': page_url,
                    'cover_image_url': post_url,
                    'repo_id': item.id,
                    'bangumi_name': bangumi_name,
                    'images': meta['total'],
                    'clusters': len([x for x in meta['ids'] if x != -1]),
                    'suffix': suffix,
                    'updated_at': last_modified.timestamp(),
                })
                total_images += meta['total']
                total_clusters += len([x for x in meta['ids'] if x != -1])
                total_animes += 1

                # This is a data hub utilized by the [DeepGHS team](https://huggingface.co/deepghs) for processing
                # anime series (in video format, including TV, OVA, movies, etc.).
            print(textwrap.dedent(f"""
                ---
                title: README
                emoji: 🌖
                colorFrom: green
                colorTo: red
                sdk: static
                pinned: false
                ---

                ## What is this?

                After downloading anime videos to our GPU cluster, we employ various computer vision algorithms to 
                extract frames, crop, and **cluster them based on character features**. These processed frames are 
                then uploaded here to reduce the manual sorting effort required for character images.

                The data in this repository will undergo automated secondary processing to remove noise, 
                after which it will be packaged and uploaded to [CyberHarem](https://huggingface.co/CyberHarem). 
                It will then be integrated into an automated pipeline for training character LoRA.

                ## Current Anime Database (constantly updated)

                Last updated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")},
                contains {plural_word(total_animes, "anime")}, {plural_word(total_images, "image")} 
                and {plural_word(total_clusters, "cluster")} in total.
            """).strip(), file=f)

            rows = sorted(rows, key=lambda x: dateparser.parse(x['Last Modified']), reverse=True)
            df = pd.DataFrame(rows)
            print(df.to_markdown(index=False), file=f)

            df_data = pd.DataFrame(data_rows)
            df_data.to_parquet(os.path.join(td, 'data.parquet'), engine='pyarrow', index=False)

        operations = []
        for directory, _, files in os.walk(td):
            for file in files:
                filename = os.path.abspath(os.path.join(directory, file))
                relpath = os.path.relpath(filename, td)
                operations.append(CommitOperationAdd(
                    path_in_repo=relpath,
                    path_or_fileobj=filename,
                ))

        current_time = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
        commit_message = f'Update lfs images, on {current_time}'
        logging.info(f'Updating lfs images to repository {repository!r} ...')
        hf_client.create_commit(
            repository,
            operations,
            commit_message=commit_message,
            repo_type='space',
            revision='main',
        )
