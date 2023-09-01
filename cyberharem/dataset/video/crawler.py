import logging
import os.path
import re
import zipfile
from typing import Optional, Union, List

from hbutils.system import TemporaryDirectory
from huggingface_hub import hf_hub_url
from unidecode import unidecode
from waifuc.source import EmptySource, LocalSource

from ..crawler import crawl_dataset_to_huggingface
from ...utils import download_file


def crawl_base_to_huggingface(
        source_repository: str, ch_id: Union[int, List[int]],
        name: str, repository: Optional[str] = None,
        limit: Optional[int] = 200, min_images: int = 10,
        no_r18: bool = False, bg_color: str = 'white', drop_multi: bool = True,
        repo_type: str = 'dataset', revision: str = 'main', path_in_repo: str = '.',
        skip_preprocess: bool = False, parallel: bool = True,
):
    ch_ids = [ch_id] if isinstance(ch_id, int) else ch_id
    source = EmptySource()
    if not repository:
        repository = 'CyberHarem/' + re.sub(r'[\W_]+', '_', unidecode(name.lower())).strip('_') + \
                     '_' + source_repository.split('/')[-1]
    logging.info(f'Target repository name {repository!r} will be used.')
    with TemporaryDirectory() as td:
        for cid in ch_ids:
            url = hf_hub_url(source_repository, filename=f'{cid}/dataset.zip', repo_type='dataset')
            os.makedirs(os.path.join(td, str(cid)), exist_ok=True)
            zip_file = os.path.join(td, str(cid), 'dataset.zip')
            download_file(url, zip_file)

            source_dir = os.path.join(td, str(cid), 'source')
            os.makedirs(source_dir, exist_ok=True)
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(source_dir)

            new_source = LocalSource(source_dir, shuffle=True)
            if parallel:
                source = source | new_source
            else:
                source = source + new_source

        return crawl_dataset_to_huggingface(
            source, repository, name,
            limit, min_images, no_r18, bg_color, drop_multi, skip_preprocess,
            repo_type, revision, path_in_repo
        )
