import glob
import io
import json
import logging
import math
import os.path
import shutil
from typing import List

import requests
from discord_webhook import DiscordWebhook
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import download_archive_as_directory
from imgutils.validate import anime_bangumi_char
from tqdm.auto import tqdm
from waifuc.action import FilterAction
from waifuc.model import ImageItem
from waifuc.source import LocalSource

from ...utils import get_hf_fs, download_file
from ...utils.ghaction import GithubActionClient


class BangumiCharTypeFilterAction(FilterAction):
    def __init__(self, types: List[str]):
        self.types = types

    def check(self, item: ImageItem) -> bool:
        type_, score = anime_bangumi_char(item.image)
        return type_ in self.types


def send_discord_publish_to_github_action(repository: str):
    client = GithubActionClient()
    client.create_workflow_run(
        'deepghs/cyberharem',
        'DC BangumiBase Publish',
        data={
            'repository': repository,
        }
    )


def publish_to_discord(repository: str, max_cnt: int = 30):
    hf = get_hf_fs()
    meta_info = json.loads(hf.read_text(f'datasets/{repository}/meta.json'))

    ids = meta_info['ids']
    char_list = []
    for id_ in tqdm(ids, desc='Check Clusters'):
        if id_ >= 0:
            filesize = hf.size(f'datasets/{repository}/{id_}/dataset.zip')
            char_list.append((id_, filesize))

    char_list = sorted(char_list, key=lambda x: (-x[1], x[0]))
    with TemporaryDirectory() as std:
        image_files = []
        for id_, _ in char_list:
            with TemporaryDirectory() as td:
                origin_dir = os.path.join(td, 'origin')
                os.makedirs(origin_dir, exist_ok=True)
                download_archive_as_directory(
                    repo_id=repository,
                    file_in_repo=f'{id_}/dataset.zip',
                    local_directory=origin_dir,
                    repo_type='dataset',
                )

                for gt in [['face'], ['halfbody'], ['imagery']]:
                    with TemporaryDirectory() as etd:
                        LocalSource(origin_dir, shuffle=True).attach(
                            BangumiCharTypeFilterAction(gt),
                        )[:1].export(etd)
                        if glob.glob(os.path.join(etd, '*.png')):
                            src_file = glob.glob(os.path.join(etd, '*.png'))[0]
                            dst_file = os.path.join(std, os.path.basename(src_file))
                            shutil.copyfile(src_file, dst_file)
                            image_files.append(dst_file)

                    if len(image_files) >= max_cnt:
                        break

            if len(image_files) >= max_cnt:
                break

        hf_url = f'https://huggingface.co/datasets/{repository}'
        bangumi_name = meta_info['name']
        logging.info(f'Getting post url for {bangumi_name!r} ...')
        from .bangumibase import get_animelist_info
        page_url, post_url = get_animelist_info(bangumi_name)
        if post_url:
            post_file = os.path.join(std, 'posts', f'{bangumi_name}.jpg')
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

        with io.StringIO() as sio:
            print(f"Screencap collection of `{bangumi_name}` has been published "
                  f"to huggingface repository: {hf_url}.", file=sio)

            if page_url:
                print(f"* MyAnimeList link of this anime: {page_url}", file=sio)
            print(f"* {plural_word(len(char_list), 'character')} were auto-clustered.", file=sio)
            print(f"* {plural_word(meta_info['total'], 'images')} were auto-extracted.", file=sio)

            webhook = DiscordWebhook(
                url=os.environ['DC_BANGUMIBASE_WEBHOOK'],
                content=sio.getvalue(),
            )
            if post_file:
                with open(post_file, 'rb') as f:
                    webhook.add_file(file=f.read(), filename=os.path.basename(post_file))
            webhook.execute()

        batch_size = 10
        for batch_id in range(int(math.ceil(len(image_files) / batch_size))):
            webhook = DiscordWebhook(
                url=os.environ['DC_BANGUMIBASE_WEBHOOK'],
                content=f'{plural_word(len(image_files), "image")} for bangumi `{bangumi_name}`\'s preview'
                if batch_id == 0 else '',
            )
            batch_image_files = image_files[batch_size * batch_id: batch_size * (batch_id + 1)]
            for img_file in batch_image_files:
                with open(img_file, 'rb') as f:
                    webhook.add_file(file=f.read(), filename=os.path.basename(img_file))
            webhook.execute()
