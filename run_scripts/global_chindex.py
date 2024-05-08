import json
import logging
import os
from functools import lru_cache
from typing import List, Dict, Iterator

import numpy as np
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import download_file_to_file, upload_directory_as_directory
from huggingface_hub import HfFileSystem, hf_hub_download
from imgutils.metrics import ccip_merge
from tqdm import tqdm
from waifuc.source import DanbooruSource
from waifuc.utils import srequest

hf_fs = HfFileSystem(token=os.environ.get('HF_TOKEN'))

SRC_REPO = 'deepghs/character_index'


@lru_cache()
def _get_source_list() -> List[dict]:
    return json.loads(hf_fs.read_text(f'datasets/{SRC_REPO}/characters.json'))


@lru_cache()
def _get_source_dict() -> Dict[str, dict]:
    return {item['tag']: item for item in _get_source_list()}


def list_character_tags() -> Iterator[str]:
    for item in _get_source_list():
        yield item['tag']


def get_detailed_character_info(tag: str) -> dict:
    return _get_source_dict()[tag]


def get_np_feats(tag, use_cache: bool = False):
    item = get_detailed_character_info(tag)
    if not use_cache:
        with TemporaryDirectory() as td:
            np_file = os.path.join(td, 'feat.npy')
            download_file_to_file(
                repo_id=SRC_REPO,
                repo_type='dataset',
                file_in_repo=f'{item["hprefix"]}/{item["short_tag"]}/feat.npy',
                local_file=np_file,
            )
            return np.load(np_file)
    else:
        return np.load(hf_hub_download(
            repo_id=SRC_REPO,
            repo_type='dataset',
            filename=f'{item["hprefix"]}/{item["short_tag"]}/feat.npy',
        ))


@lru_cache()
def _db_session():
    s = DanbooruSource(['1girl'])
    s._prune_session()
    return s.session


def _get_consequent_tags(tag) -> List[str]:
    session = _db_session()
    resp = srequest(session, 'GET', 'https://danbooru.donmai.us/tag_implications.json', params={
        'search[implied_from]': tag
    })
    return [item['consequent_name'] for item in resp.json()]


def _get_antecedent_tags(tag) -> List[str]:
    session = _db_session()
    resp = srequest(session, 'GET', 'https://danbooru.donmai.us/tag_implications.json', params={
        'search[implied_to]': tag
    })
    return [item['antecedent_name'] for item in resp.json()]


def runit():
    tags = list(list_character_tags())
    tag_set = set(tags)

    embs = []
    tag_infos = []
    for tag in tqdm(tags):
        logging.info(f'Checking tag {tag!r} ...')
        ft = ccip_merge(get_np_feats(tag, use_cache=True))
        ft = ft / np.linalg.norm(ft)
        info = get_detailed_character_info(tag)
        consequent_tags = _get_consequent_tags(tag)
        if not any(t in tag_set for t in consequent_tags):
            info['consequent_tags'] = consequent_tags
            info['antecedent_tags'] = _get_antecedent_tags(tag)
            embs.append(ft)
            tag_infos.append(info)
        else:
            ptags = [t for t in consequent_tags if t in tag_set]
            logging.info(f'Tag {tag!r} is consequent of {ptags!r}, skipped.')

    with TemporaryDirectory() as td:
        embeddings = np.stack(embs)
        np.save(os.path.join(td, 'embeddings'), embeddings)
        with open(os.path.join(td, 'tag_infos.json'), 'w') as f:
            json.dump(tag_infos, f, indent=4, sort_keys=True, ensure_ascii=False)

        upload_directory_as_directory(
            repo_id=SRC_REPO,
            repo_type='dataset',
            local_directory=td,
            path_in_repo='index',
            message=f'Making index, containing {plural_word(len(tag_infos), "character tag")}'
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    runit()
