import datetime
import os.path
import re
import time
import warnings
from dataclasses import asdict
from functools import lru_cache
from urllib.parse import urljoin, quote_plus

import numpy as np
import pandas as pd
import requests
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory, urlsplit
from hfutils.operate import upload_directory_as_directory
from huggingface_hub import hf_hub_url
from pyanimeinfo.myanimelist import JikanV4Client
from pynyaasi.nyaasi import NyaaSiClient
from pyquery import PyQuery as pq
from requests import HTTPError
from tqdm import tqdm

from ..dataset.video.bangumibase import hf_client
from ..utils import get_requests_session, download_file, number_to_tag, get_hf_client

session = get_requests_session()

resp = session.get('https://subsplease.org/shows/')
page = pq(resp.text)
nyaasi_client = NyaaSiClient()
jikan_client = JikanV4Client()


def _name_safe(name_text):
    return re.sub(r'[\W_]+', ' ', name_text).strip(' ')


@lru_cache()
def _get_info_mal():
    hf_client = get_hf_client()
    df = pd.read_parquet(hf_client.hf_hub_download(
        repo_id='deepghs/subsplease_mal',
        repo_type='dataset',
        filename='table.parquet'
    )).replace(np.nan, None)
    df = df[~df['mal_id'].isnull()]
    df['mal_id'] = df['mal_id'].map(int)
    return {
        item['page_id']: {
            'mal_id': item['mal_id'],
            'subsplease': {
                'batch': item['subsplease_batch'],
                'episode': item['subsplease_episode'],
            }
        } for item in df.to_dict('records')
    }


def _get_info_from_mal_repo(page_id: str):
    mal = _get_info_mal()
    if page_id in mal:
        v = mal[page_id]
        while True:
            try:
                mal_info = jikan_client.get_anime_full(v['mal_id'])
            except HTTPError as err:
                if err.response.status_code == 429:
                    warnings.warn(f'429 error detected: {err!r}, wait for some seconds ...')
                    time.sleep(5.0)
                else:
                    raise
            else:
                break
        return {**v, 'mal': mal_info}
    else:
        return None


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


def sync(repository: str):
    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', exist_ok=True)

    episodes = []
    animes = []

    for ei, show_item in tqdm(list(enumerate(page('#main .all-shows .all-shows-link').items()))[:],
                              desc='All Animes'):
        url = urljoin(resp.url, show_item('a').attr('href'))
        title = show_item('a').text().strip()
        logging.info(f'Anime {title!r}, homepage url: {url!r} ...')

        assert urlsplit(url).path_segments[1] == 'shows'
        page_id = urlsplit(url).path_segments[2]

        full_item = _get_info_from_mal_repo(page_id)
        if not full_item:
            continue
        myanime_item = full_item['mal']
        if not myanime_item:
            continue

        collected_nyaasi_ids = set()
        for episode_item in full_item['subsplease']['episode']:
            for ditem in episode_item['downloads']:
                torrent_url = ditem['torrent']
                assert urlsplit(torrent_url).path_segments[1] == 'view'
                nyaasi_id = int(urlsplit(torrent_url).path_segments[2])
                collected_nyaasi_ids.add(nyaasi_id)

        search_query_text = f'subsplease {_name_safe(title)} 1080p mkv'
        logging.info(f'Searching from nyaasi with query text - {search_query_text!r} ...')
        vs = []
        for item in nyaasi_client.iter_items(search_query_text):
            ditem = asdict(item)
            if ditem['id'] in collected_nyaasi_ids:
                if ditem['category']:
                    ditem['category'] = ditem['category'].name
                vs.append({'anime_id': myanime_item['mal_id'], **ditem})
        vdf = pd.DataFrame(vs)
        logging.info(f'{plural_word(len(vdf), "episode")} found in nyaasi')
        if not len(vdf) > 0:
            logging.warning('No episodes found, skipped.')
            continue
        aitem = {
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

            'subsplease_title': title,
            'subsplease_url': url,
            'nyaasi_episodes': len(vdf),
            'nyaasi_seeders_std75': np.percentile(vdf['seeders'], 25).item(),
            'nyaasi_seeders_avg': np.mean(vdf['seeders']).item(),
            'nyaasi_downloads_avg': np.mean(vdf['downloads']).item(),
        }
        episodes.extend(vs)
        animes.append(aitem)

    df_episodes = pd.DataFrame(episodes)
    df_animes = pd.DataFrame(animes)
    df_animes = df_animes.sort_values(by=['nyaasi_seeders_std75', 'id'], ascending=[False, False])
    df_animes = df_animes.replace(np.NaN, None)

    logging.info(f'Episodes:\n{df_episodes}')
    logging.info(f'Animes:\n{df_animes}')

    df_animes_x = df_animes[(df_animes['nyaasi_seeders_std75'] >= 15) & ~df_animes['airing']]
    df_animes_x = df_animes_x.sort_values(by=['nyaasi_downloads_avg', 'id'], ascending=[False, False])
    df_animes_xs = df_animes_x[['id', 'title', 'episodes', 'nyaasi_episodes', 'airing', 'status',
                                'nyaasi_seeders_std75', 'nyaasi_downloads_avg']]
    logging.info(f'Available animes:\n{df_animes_xs}')

    with TemporaryDirectory() as td:
        episodes_file = os.path.join(td, 'episodes.parquet')
        df_episodes.to_parquet(episodes_file, engine='pyarrow', index=False)

        animes_file = os.path.join(td, 'animes.parquet')
        df_animes.to_parquet(animes_file, engine='pyarrow', index=False)

        images_dir = os.path.join(td, 'images')
        os.makedirs(images_dir, exist_ok=True)
        magnets_dir = os.path.join(td, 'magnets')
        os.makedirs(magnets_dir, exist_ok=True)

        l_shown = []
        for aitem in tqdm(df_animes.to_dict('records'), desc='Dump Items'):
            logging.info(f'Making item for {aitem["id"]}, title: {aitem["title"]!r} ...')
            safe_bangumi_name = aitem['subsplease_title'].replace('`', ' ').replace('[', '(').replace(']', ')')
            search_query_text = f'subsplease {_name_safe(aitem["subsplease_title"])} 1080p mkv'
            nyaasi_url = f'https://nyaa.si/?f=0&c=1_0&q={quote_plus(search_query_text)}'

            df_anime_episodes = df_episodes[df_episodes['anime_id'] == aitem['id']]
            last_updated = datetime.datetime.fromtimestamp(
                df_anime_episodes['timestamp'].max().item()).strftime('%Y-%m-%d %H:%M')

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

            magnet_file = os.path.join(magnets_dir, f'{suffix}.txt')
            with open(magnet_file, 'w') as f:
                for eitem in df_anime_episodes.to_dict('records'):
                    print(eitem['magnet_url'], file=f)
            magnet_url = hf_hub_url(repo_id=repository, repo_type='dataset', filename=os.path.relpath(magnet_file, td))

            seeders = int(round(aitem['nyaasi_seeders_std75']))
            if seeders >= 80:
                seeders_md = f'**{seeders}**'
            elif seeders >= 15:
                seeders_md = f'{seeders}'
            else:
                seeders_md = f'~{seeders}~'

            l_shown.append({
                'ID': aitem['id'],
                'Post': post_md,
                'Bangumi': f'[{safe_bangumi_name}]({aitem["subsplease_url"]})',
                'Type': aitem['type'],
                'Episodes': f'{aitem["nyaasi_episodes"]} / {int(aitem["episodes"]) if aitem["episodes"] else "?"}',
                'Status': aitem['status'] if aitem['airing'] else f'**{aitem["status"]}**',
                'Score': aitem['score'],
                'Nyaasi': f'[Search]({nyaasi_url})',
                'Magnets': f'[Download]({magnet_url})',
                'Seeds': seeders_md,
                'Downloads': int(round(aitem['nyaasi_downloads_avg'])),
                'Updated At': last_updated,
            })
        df_l_shown = pd.DataFrame(l_shown)
        df_l_shown = df_l_shown.replace(np.nan, 'N/A')

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
            print('- nyaasi', file=f)
            print('- subsplease', file=f)
            print('---', file=f)
            print('', file=f)

            print('This is an integration database of subsplease, myanimelist and nyaasi. '
                  'You can know which animes are the hottest ones currently, '
                  'and which of them have well-seeded magnet links.', file=f)
            print('', file=f)
            print('This database is refreshed daily.', file=f)
            print('', file=f)

            print(f'## Current Animes', file=f)
            print('', file=f)
            current_time = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
            print(f'{plural_word(len(df_animes), "anime")}, '
                  f'{plural_word(len(df_episodes), "episode")} in total, '
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
            message=f'Sync {plural_word(len(df_animes), "anime")}, '
                    f'with {plural_word(len(df_episodes), "episode")}',
            clear=True,
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    sync(
        repository='deepghs/subsplease_animes',
    )
