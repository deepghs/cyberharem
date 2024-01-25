import glob
import json
import logging
import os.path
import re
import zipfile
from typing import Optional, Union, List

from hbutils.system import TemporaryDirectory
from huggingface_hub import hf_hub_url
from unidecode import unidecode
from waifuc.action import CCIPAction, FilterSimilarAction, RandomFilenameAction, TaggingAction
from waifuc.source import EmptySource, LocalSource

from ..crawler import crawl_dataset_to_huggingface
from ...utils import download_file, get_hf_fs, get_global_namespace


def crawl_base_to_huggingface(
        source_repository: str, ch_id: Union[int, List[int]],
        name: str, display_name: Optional[str] = None, repository: Optional[str] = None,
        limit: Optional[int] = 1000, min_images: int = 10,
        no_r18: bool = False, bg_color: str = 'white', drop_multi: bool = True,
        repo_type: str = 'dataset', revision: str = 'main', path_in_repo: str = '.',
        skip_preprocess: bool = True, parallel: bool = True, standalone_ccip: bool = True,
        keep_cnt_ratio: bool = True, private: bool = False,
):
    ch_ids = [ch_id] if isinstance(ch_id, int) else ch_id
    source = EmptySource()
    alphabet_name = re.sub(r'[\W_]+', '_', unidecode(name.lower())).strip('_').lower() + '_' + \
                    source_repository.split('/')[-1]
    if not repository:
        repository = f'{get_global_namespace()}/{alphabet_name}'
    logging.info(f'Target repository name {repository!r} will be used.')

    hf_fs = get_hf_fs()
    source_meta_info = json.loads(hf_fs.read_text(f'datasets/{source_repository}/meta.json'))
    bangumi_name = source_meta_info['name']
    display_name = display_name or f'{name} ({bangumi_name})'
    with TemporaryDirectory() as td:
        img_cnts = []
        for cid in ch_ids:
            url = hf_hub_url(source_repository, filename=f'{cid}/dataset.zip', repo_type='dataset')
            os.makedirs(os.path.join(td, str(cid)), exist_ok=True)
            zip_file = os.path.join(td, str(cid), 'dataset.zip')
            download_file(url, zip_file)

            source_dir = os.path.join(td, str(cid), 'source')
            os.makedirs(source_dir, exist_ok=True)
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(source_dir)
            img_cnts.append(len(glob.glob(os.path.join(source_dir, '*.png'))))

        total = sum(img_cnts)
        for cid, c_cnt in zip(ch_ids, img_cnts):
            source_dir = os.path.join(td, str(cid), 'source')
            new_source = LocalSource(source_dir, shuffle=True)
            if standalone_ccip:
                new_source = new_source.attach(CCIPAction())
            if keep_cnt_ratio:
                new_source = new_source[:int(round(c_cnt * 1.0 / total * limit))]

            if parallel:
                source = source | new_source
            else:
                source = source + new_source
            if skip_preprocess:
                source = source.attach(
                    FilterSimilarAction('all'),
                    RandomFilenameAction(ext='.png'),
                    TaggingAction(force=False, character_threshold=1.01),
                )

        return crawl_dataset_to_huggingface(
            source=source,
            repository=repository,
            name=alphabet_name,
            display_name=display_name,
            limit=limit,
            min_images=min_images,
            no_r18=no_r18,
            bg_color=bg_color,
            drop_multi=drop_multi,
            skip_preprocess=skip_preprocess,
            no_monochrome_check=False,
            repo_type=repo_type,
            revision=revision,
            path_in_repo=path_in_repo,
            private=private,
            bangumi_source_repository=source_repository,
        )
