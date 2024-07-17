import glob
import json
import logging
import os.path
import shutil
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from typing import ContextManager, Tuple

import pandas as pd
from hbutils.system import TemporaryDirectory
from huggingface_hub import hf_hub_download

from .subsplease import _name_safe


@lru_cache()
def _get_all_animes():
    return pd.read_parquet(hf_hub_download(
        repo_id='deepghs/subsplease_animes',
        repo_type='dataset',
        filename='animes.parquet'
    ))


def get_available_animes():
    df_src = _get_all_animes()
    df_dst = pd.read_parquet(hf_hub_download(
        repo_id='BangumiBase/README',
        repo_type='space',
        filename='data.parquet'
    ))

    df_src_to_download = df_src[
        (df_src['subsplease_seeders_std75'] >= 30) &
        (~df_src['airing']) &
        (df_src['subsplease_episodes'] >= 8) &
        (df_src['episodes'] >= 8) &
        (~df_src['id'].isin(df_dst['id']))
        ]
    df_src_to_download = df_src_to_download.sort_values(
        by=['subsplease_downloads_avg', 'id'], ascending=[False, False])
    return df_src_to_download


@lru_cache()
def _get_all_episodes() -> pd.DataFrame:
    return pd.read_parquet(hf_hub_download(
        repo_id='deepghs/subsplease_animes',
        repo_type='dataset',
        filename='episodes.parquet'
    ))


def get_anime_episodes(anime_id: int) -> pd.DataFrame:
    df = _get_all_episodes()
    return df[df['anime_id'] == anime_id]


@contextmanager
def mock_magnet_input_file(anime_id: int) -> ContextManager[Tuple[str, int]]:
    with TemporaryDirectory() as td:
        count = 0
        magnet_file = os.path.join(td, 'magnets.txt')
        with open(magnet_file, 'w') as f:
            for item in get_anime_episodes(anime_id).to_dict('records'):
                print(item['magnet_url'], file=f)
                count += 1

        yield magnet_file, count


@lru_cache()
def _get_anime_info(anime_id: int) -> dict:
    df = _get_all_animes()
    items = df[df['id'] == anime_id].to_dict('records')
    if not items:
        raise ValueError(f'Anime {anime_id!r} not found.')
    else:
        return items[0]


def get_anime_token(anime_id: int) -> str:
    item = _get_anime_info(anime_id)
    return f'{item["id"]}__{_name_safe(item["title"]).lower().replace(" ", "_")}'


_ANIME_ROOT = 'animes'


def _get_anime_workspace(anime_id: int) -> str:
    return os.path.abspath(os.path.join(_ANIME_ROOT, get_anime_token(anime_id)))


def _init_anime_workspace(anime_id: int):
    item = _get_anime_info(anime_id)
    workspace = _get_anime_workspace(anime_id)
    os.makedirs(workspace, exist_ok=True)
    with open(os.path.join(workspace, 'meta.json'), 'w') as f:
        json.dump(item, f, indent=4, sort_keys=True, ensure_ascii=False)

    videos_dir = os.path.join(workspace, 'videos')
    os.makedirs(videos_dir, exist_ok=True)

    return workspace


def get_workspace_info(anime_id: int):
    workspaces = glob.glob(os.path.join(_ANIME_ROOT, f'{anime_id}__*'))
    if workspaces:
        assert len(workspaces) == 1, f'Not unique workspaces - {workspaces!r}.'
        workspace = workspaces[0]
    else:
        workspace = _init_anime_workspace(anime_id)

    with open(os.path.join(workspace, 'meta.json'), 'r') as f:
        meta = json.load(f)

    if os.path.exists(os.path.join(workspace, 'status.json')):
        with open(os.path.join(workspace, 'status.json'), 'r') as f:
            status = json.load(f)['status']
    else:
        status = 'pending'

    return workspace, meta, status


_ARIA2C = shutil.which('aria2c')


def download_anime_videos(anime_id: int):
    workspace, meta, status = get_workspace_info(anime_id)
    if status == 'pending':  # need downloading
        if not _ARIA2C:
            raise EnvironmentError('No aria2c found, you can install by with `apt install aria2` on ubuntu.')

        with mock_magnet_input_file(anime_id) as (magnet_file, magnet_count):
            commands = [_ARIA2C, '--seed-time=0', '-i', magnet_file, '-j', str(magnet_count)]
            cwd = os.path.join(workspace, 'videos')
            os.makedirs(cwd, exist_ok=True)
            logging.info(f'Downloading anime {anime_id!r} ({meta["title"]!r}) with '
                         f'command {commands!r}, workdir: {cwd!r} ...')
            terminal_size = os.get_terminal_size()
            process = subprocess.run(
                commands, cwd=cwd,
                env={
                    **os.environ,
                    'COLUMNS': str(terminal_size.columns),
                    'LINES': str(terminal_size.lines),
                },
                bufsize=0,
            )
            process.check_returncode()

            with open(os.path.join(workspace, 'status.json'), 'w') as f:
                json.dump({
                    'status': 'downloaded',
                }, f, ensure_ascii=False, sort_keys=True, indent=4)
            logging.info('Download complete!')


    else:
        logging.info(f'Anime {anime_id!r} ({meta["title"]!r}) already downloaded, skipped.')