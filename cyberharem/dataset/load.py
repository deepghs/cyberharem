import logging
import os.path
import zipfile
from contextlib import contextmanager
from typing import ContextManager, Tuple, Optional, Union

from gchar.games import get_character
from gchar.games.base import Character
from hbutils.system import TemporaryDirectory, urlsplit
from huggingface_hub import hf_hub_url
from waifuc.utils import download_file

from ..utils import get_hf_fs, get_ch_name


@contextmanager
def load_dataset_for_character(source, size: Union[Tuple[int, int], str] = (512, 704)) \
        -> ContextManager[Tuple[Optional[Character], str]]:
    if isinstance(source, str) and os.path.exists(source):
        if os.path.isdir(source):
            logging.info(f'Dataset directory {source!r} loaded.')
            yield None, source
        elif os.path.isfile(source):
            with zipfile.ZipFile(source, 'r') as zf, TemporaryDirectory() as td:
                zf.extractall(td)
                logging.info(f'Archive dataset {source!r} unzipped to {td!r} and loaded.')
                yield None, td
        else:
            raise OSError(f'Unknown local source - {source!r}.')

    else:
        if isinstance(source, Character):
            repo = f'CyberHarem/{get_ch_name(source)}'
        else:
            try_ch = get_character(source)
            if try_ch is None:
                repo = source
            else:
                source = try_ch
                repo = f'CyberHarem/{get_ch_name(source)}'

        hf_fs = get_hf_fs()
        if isinstance(size, tuple):
            width, height = size
            ds_name = f'{width}x{height}'
        elif isinstance(size, str):
            ds_name = size
        else:
            raise TypeError(f'Unknown dataset type - {size!r}.')
        if hf_fs.exists(f'datasets/{repo}/dataset-{ds_name}.zip'):
            logging.info(f'Online dataset {repo!r} founded.')
            zip_url = hf_hub_url(repo_id=repo, repo_type='dataset', filename=f'dataset-{ds_name}.zip')
            with TemporaryDirectory() as dltmp:
                zip_file = os.path.join(dltmp, 'dataset.zip')
                download_file(zip_url, zip_file, desc=f'{repo}/{urlsplit(zip_url).filename}')

                with zipfile.ZipFile(zip_file, 'r') as zf, TemporaryDirectory() as td:
                    zf.extractall(td)
                    logging.info(f'Online dataset {repo!r} loaded at {td!r}.')
                    yield source, td

        else:
            raise ValueError(f'Remote dataset {repo!r} not found for {source!r}.')
