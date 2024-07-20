import glob
import json
import mimetypes
import os.path
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from functools import lru_cache, partial
from typing import ContextManager, Tuple

import click
import pandas as pd
from ditk import logging
from gchar.utils import print_version as _origin_print_version
from hbutils.scale import time_to_delta_str
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from unidecode import unidecode

from cyberharem.utils.cli import GLOBAL_CONTEXT_SETTINGS
from .subsplease import _name_safe
from ..utils import get_global_bg_namespace


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
        (df_src['nyaasi_seeders_std75'] >= 30) &
        (~df_src['airing']) &
        (df_src['nyaasi_episodes'] >= 8) &
        (df_src['episodes'] >= 8) &
        (~df_src['id'].isin(df_dst['id']))
        ]
    df_src_to_download = df_src_to_download.sort_values(
        by=['nyaasi_downloads_avg', 'id'], ascending=[False, False])
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
def mock_magnet_input_file(anime_id: int, min_seeders: int = 10) -> ContextManager[Tuple[str, int]]:
    with TemporaryDirectory() as td:
        count = 0
        magnet_file = os.path.join(td, 'magnets.txt')
        with open(magnet_file, 'w') as f:
            for item in get_anime_episodes(anime_id).to_dict('records'):
                if item['seeders'] >= min_seeders:
                    print(item['magnet_url'], file=f)
                    count += 1
                else:
                    logging.warning(f'Resource {item["title"]!r} has too few seeders ({item["seeders"]}), skipped.')

        if count < 4:
            raise ValueError(f'Magnet count too few - {count!r}.')
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


def download_anime_videos(anime_id: int, min_video_files: int = 4, seed_minutes: int = 10):
    workspace, meta, status = get_workspace_info(anime_id)
    if status == 'pending':  # need downloading
        if not _ARIA2C:
            raise EnvironmentError('No aria2c found, you can install by with `apt install aria2` on ubuntu.')

        with mock_magnet_input_file(anime_id) as (magnet_file, magnet_count):
            commands = [_ARIA2C, f'--seed-time={seed_minutes}', '-i', magnet_file, '-j', str(magnet_count)]
            cwd = os.path.join(workspace, 'videos')
            os.makedirs(cwd, exist_ok=True)
            logging.info(f'Downloading anime {anime_id!r} ({meta["title"]!r}) with '
                         f'command {commands!r}, workdir: {cwd!r} ...')
            terminal_size = os.get_terminal_size()
            start_time = time.time()
            process = subprocess.run(
                commands, cwd=cwd,
                env={
                    **os.environ,
                    'COLUMNS': str(terminal_size.columns),
                    'LINES': str(terminal_size.lines),
                },
                bufsize=0,
            )
            download_duration = time.time() - start_time
            download_minutes = int(max(round(download_duration / 60.0), 1))
            logging.info(f'Download duration: {time_to_delta_str(download_duration)}')
            if process.returncode != 0:
                if glob.glob(os.path.join(cwd, '**', '*.aria2'), recursive=True):
                    raise ChildProcessError(f'Uncompleted download at {cwd!r}, exitcode {process.returncode}.')
                else:
                    logging.warning(f'Completed download, but exit {process.returncode}.')

            # seed_command = [_ARIA2C, f'--seed-time={download_minutes}', '-i', magnet_file, '-j', str(magnet_count)]
            # logging.info(f'Seeding resource for {plural_word(download_minutes, "minute")}, '
            #              f'with command: {seed_command!r} ...')
            # devnull = open(os.devnull, 'w')
            # process = subprocess.Popen(
            #     seed_command,
            #     stdout=devnull,
            #     stderr=devnull,
            #     start_new_session=True,
            #     cwd=cwd,
            # )
            # logging.info(f'Seeding process started, pid: {process} ...')

            video_files = []
            for root, _, files in os.walk(cwd):
                for file in files:
                    mimetype, _ = mimetypes.guess_type(file)
                    if mimetype.startswith('video/'):
                        video_files.append(file)
            if len(video_files) < min_video_files:
                raise ValueError(f'Too few video files - {video_files!r}.')
            else:
                logging.info(f'{plural_word(len(video_files), "video file")} found in {cwd!r}.')

            with open(os.path.join(workspace, 'status.json'), 'w') as f:
                json.dump({
                    'status': 'downloaded',
                }, f, ensure_ascii=False, sort_keys=True, indent=4)
            logging.info('Download complete!')

    else:
        logging.info(f'Anime {anime_id!r} ({meta["title"]!r}) already downloaded, skipped.')


def make_bangumibase(anime_id, force_remake: bool = False, min_size: int = 320, no_extract: bool = False,
                     max_images_limit: int = 50000, all_frames: bool = False, ):
    logging.info(f'Try downloading {anime_id!r} ...')
    download_anime_videos(anime_id)

    from ..dataset.video.extract import extract_to_huggingface

    workspace, meta, status = get_workspace_info(anime_id)
    if not force_remake and status == 'completed':
        logging.info(f'Anime {anime_id!r} already maked, skipped.')
        return

    logging.info(f'Workspace for anime {anime_id!r}: {workspace}')
    bangumi_name = meta['title']
    rname = re.sub(r'[\W_]+', '', unidecode(bangumi_name.lower()))
    repository = f"{get_global_bg_namespace()}/{rname}"
    logging.info(f'Bangumi name: {bangumi_name!r}, repository: {repository!r}.')
    videos_dir = os.path.join(workspace, 'videos')
    extract_to_huggingface(
        video_or_directory=videos_dir,
        bangumi_name=bangumi_name,
        repository=repository,
        revision='main',
        no_extract=no_extract,
        min_size=min_size,
        max_images_limit=max_images_limit,
        all_frames=all_frames,
        myanimelist_id=meta['id'],
    )
    with open(os.path.join(workspace, 'status.json'), 'w') as f:
        json.dump({
            'status': 'completed',
        }, f, ensure_ascii=False, sort_keys=True, indent=4)
    logging.info('Extraction complete!')


def prepare_task_list():
    df = get_available_animes()
    df = df.sort_values(by=['nyaasi_seeders_std75', 'id'], ascending=[True, True])
    anime_ids = []
    for item in tqdm(df.to_dict('records'), desc='Preparing'):
        logging.info(f'Preparing for {item["id"]!r} ({item["title"]!r}) ...')
        workspace, meta, status = get_workspace_info(item['id'])
        if status != 'completed':
            anime_ids.append(item['id'])
        else:
            logging.info(f'Anime {item["id"]} already completed, skipped.')

    _task_list_file = os.path.join(_ANIME_ROOT, 'task_list.json')
    if os.path.dirname(_task_list_file):
        os.makedirs(os.path.dirname(_task_list_file), exist_ok=True)
    logging.info(f'Saving to task list {_task_list_file} ...')
    with open(_task_list_file, 'w') as f:
        json.dump(anime_ids, f, ensure_ascii=False, sort_keys=True, indent=4)


def prepare_refresh_list():
    _task_list_file = os.path.join(_ANIME_ROOT, 'task_list.json')
    if os.path.exists(_task_list_file):
        with open(_task_list_file, 'r') as f:
            anime_ids = json.load(f)
    else:
        anime_ids = []

    retval = []
    for anime_id in tqdm(anime_ids, desc='Preparing'):
        logging.info(f"Preparing for {anime_id!r} ...")
        workspace, meta, status = get_workspace_info(anime_id)
        if status != 'completed':
            retval.append(anime_id)
        else:
            logging.info(f"Anime {anime_id} already completed, skipped.")

    if os.path.dirname(_task_list_file):
        os.makedirs(os.path.dirname(_task_list_file), exist_ok=True)
    logging.info(f'Saving to task list {_task_list_file} ...')
    with open(_task_list_file, 'w') as f:
        json.dump(retval, f, ensure_ascii=False, sort_keys=True, indent=4)


def get_task_list():
    _task_list_file = os.path.join(_ANIME_ROOT, 'task_list.json')
    logging.info(f'Loading task list {_task_list_file} ...')
    with open(_task_list_file, 'r') as f:
        return json.load(f)


print_version = partial(_origin_print_version, 'cyberharem.info.video')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Progress anime info')
@click.option('-v', '--version', is_flag=True, callback=print_version, expose_value=False, is_eager=True)
def cli():
    pass  # pragma: no cover


@cli.command('list', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Make task list')
def list_():
    logging.try_init_root(logging.INFO)
    prepare_task_list()


@cli.command('relist', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Make task list')
def relist_():
    logging.try_init_root(logging.INFO)
    prepare_refresh_list()


@cli.command('extract', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Make task list')
def extract():
    logging.try_init_root(logging.INFO)
    world_size = int(os.environ['CH_WORLD_SIZE'])
    rank = int(os.environ['CH_RANK'])
    logging.info(f'World size: {world_size}, rank: {rank}.')
    anime_ids = get_task_list()[rank::world_size]
    logging.info(f'{plural_word(len(anime_ids), "anime")} in total.')

    for anime_id in tqdm(anime_ids, desc='Extract Animes'):
        logging.info(f'Try downloading {anime_id!r} ...')
        try:
            download_anime_videos(anime_id)
        except ValueError as err:
            logging.info(f'Download error - {err!r}.')
        make_bangumibase(anime_id, all_frames=True, max_images_limit=35000)


if __name__ == '__main__':
    cli()
