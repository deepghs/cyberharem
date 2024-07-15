import datetime
import os.path
import re
from dataclasses import asdict
from urllib.parse import urljoin, quote_plus

import numpy as np
import pandas as pd
import requests
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory
from huggingface_hub import hf_hub_url
from pyanimeinfo.myanimelist import JikanV4Client
from pynyaasi.nyaasi import NyaaSiClient
from pyquery import PyQuery as pq
from thefuzz import fuzz
from tqdm import tqdm

from cyberharem.dataset.video.bangumibase import hf_client
from cyberharem.utils import get_requests_session, download_file, number_to_tag

logging.try_init_root(logging.INFO)
session = get_requests_session()

resp = session.get('https://subsplease.org/shows/')
page = pq(resp.text)
nyaasi_client = NyaaSiClient()
jikan_client = JikanV4Client()


def _name_safe(name_text):
    return re.sub(r'[\W_]+', ' ', name_text).strip(' ')


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


if __name__ == '__main__':
    repository = 'deepghs/subsplease_animes'
    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset')

    episodes = []
    animes = []

    for ei, show_item in tqdm(list(enumerate(page('#main .all-shows .all-shows-link').items()))[:],
                              desc='All Animes'):
        url = urljoin(resp.url, show_item('a').attr('href'))
        title = show_item('a').text().strip()
        logging.info(f'Anime {title!r}, homepage url: {url!r} ...')

        myanime_item = None
        logging.info('Search information from myanimelist ...')
        for pyaitem in jikan_client.search_anime(query=title):
            if pyaitem['type'].lower() not in ['tv', 'movie', 'ova', 'ona']:
                continue

            max_ratio = max(
                fuzz.partial_ratio(
                    _name_safe(title).lower(),
                    _name_safe(title_item['title']).lower(),
                ) for title_item in pyaitem['titles']
            ) / 100.0
            if max_ratio > 0.9:
                myanime_item = pyaitem
                break

        if myanime_item:
            logging.info(f'Found on myanimelist: {myanime_item["url"]!r} ...')
        else:
            logging.warning('No information found on myanime list, skipped.')
            continue

        vs = []
        search_query_text = f'subsplease {_name_safe(title)} 1080p mkv'
        logging.info(f'Searching from nyaasi with query text - {search_query_text!r} ...')
        for item in nyaasi_client.iter_items(search_query_text):
            ditem = asdict(item)
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
            'subsplease_episodes': len(vdf),
            'subsplease_seeders_std75': np.percentile(vdf['seeders'], 25).item(),
            'subsplease_seeders_avg': np.mean(vdf['seeders']).item(),
            'subsplease_downloads_avg': np.mean(vdf['downloads']).item(),
        }
        episodes.extend(vs)
        animes.append(aitem)

    df_episodes = pd.DataFrame(episodes)
    df_animes = pd.DataFrame(animes)
    df_animes = df_animes.sort_values(by=['subsplease_seeders_std75', 'id'], ascending=[False, False])
    df_animes = df_animes.replace(np.NaN, None)

    logging.info(f'Episodes:\n{df_episodes}')
    logging.info(f'Animes:\n{df_animes}')

    df_animes_x = df_animes[(df_animes['subsplease_seeders_std75'] >= 15) & ~df_animes['airing']]
    df_animes_x = df_animes_x.sort_values(by=['subsplease_downloads_avg', 'id'], ascending=[False, False])
    df_animes_xs = df_animes_x[['id', 'title', 'episodes', 'subsplease_episodes', 'airing', 'status',
                                'subsplease_seeders_std75', 'subsplease_downloads_avg']]
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
            safe_bangumi_name = aitem['title'].replace('`', ' ').replace('[', '(').replace(']', ')')
            search_query_text = f'subsplease {_name_safe(aitem["subsplease_title"])} 1080p mkv'
            nyaasi_url = f'https://nyaa.si/?f=0&c=1_0&q={quote_plus(search_query_text)}'

            df_anime_episodes = df_episodes[df_episodes['anime_id'] == aitem['id']]
            last_updated = datetime.datetime.fromtimestamp(
                df_anime_episodes['timestamp'].max().item()).strftime('%Y-%m-%d %H:%M')

            suffix = _name_safe(aitem['title']).replace(' ', '_').lower()
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

            seeders = int(round(aitem['subsplease_seeders_std75']))
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
                'Episodes': f'{aitem["subsplease_episodes"]} / {int(aitem["episodes"]) if aitem["episodes"] else "?"}',
                'Status': aitem['status'] if aitem['airing'] else f'**{aitem["status"]}**',
                'Nyaasi': f'[Search]({nyaasi_url})',
                'Magnets': f'[Download]({magnet_url})',
                'Seeders': seeders_md,
                'Downloads': int(round(aitem['subsplease_downloads_avg'])),
                'Last Updated': last_updated,
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
            print(f'- {number_to_tag(len(df_l_shown))}', file=f)
            print('source_datasets:', file=f)
            print('- myanimelist', file=f)
            print('- nyaasi', file=f)
            print('- subsplease', file=f)
            print('---', file=f)
            print('', file=f)

            print(f'{plural_word(len(df_l_shown), "anime")}, '
                  f'{plural_word(len(df_episodes), "episode")} in total.', file=f)
            print('', file=f)
            print(df_l_shown.to_markdown(index=False), file=f)
            print('', file=f)

        os.system(f'tree {td!r}')

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=td,
            path_in_repo='.',
            message=f'Sync {plural_word(len(df_l_shown), "anime")}, '
                    f'with {plural_word(len(df_episodes), "episode")}',
        )
