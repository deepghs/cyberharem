import logging
import os
import pathlib
import re
from contextlib import contextmanager
from typing import Tuple, ContextManager, Optional, List

from filelock import FileLock
from hfutils.operate import download_archive_as_directory
from hfutils.operate.base import RepoTypeTyping


def _reg_root_dir() -> str:
    return os.environ['REG_HOME']


def _get_reg_dir(reg_name: str, repo_id: str, archive_file: str, repo_type: RepoTypeTyping = 'dataset') \
        -> Tuple[str, str]:
    _reg_dir = os.path.join(_reg_root_dir(), reg_name)
    _reg_ds_dir = os.path.join(_reg_dir, 'dataset')
    _reg_ds_cache = os.path.join(_reg_dir, 'dataset.cache')

    _reg_lock = os.path.join(_reg_dir, '.lock')
    logging.info(f'Acquiring lock file {_reg_lock} ...')
    with FileLock(_reg_lock):
        _reg_ok_flag = os.path.join(_reg_dir, '.ok')
        if not os.path.exists(_reg_ok_flag):
            os.makedirs(_reg_ds_dir, exist_ok=True)
            logging.info(f'Downloading regularization dataset {archive_file!r} from {repo_id!r} ...')
            download_archive_as_directory(
                repo_id=repo_id,
                repo_type=repo_type,
                file_in_repo=archive_file,
                local_directory=_reg_ds_dir,
            )
            pathlib.Path(_reg_ok_flag).touch()

    return _reg_ds_dir, _reg_ds_cache


_DEFAULT_REG = ('deepghs/reg_experiment', 'all_webp.zip')


@contextmanager
def get_default_reg_dir() -> ContextManager[Tuple[str, str]]:
    repo_id, archive_file = _DEFAULT_REG
    logging.info('Loading default regularization dataset ...')
    yield _get_reg_dir('generic', repo_id, archive_file, repo_type='dataset')


@contextmanager
def get_bangumi_reg_dir(bangumi_repo_id: str, select: str = 'normal') -> ContextManager[Tuple[str, str]]:
    name = re.sub(r'[\W_]+', '_', bangumi_repo_id).strip('_')
    logging.info(f'Loading regularization dataset for bangumi {bangumi_repo_id!r} ...')
    yield _get_reg_dir(name, bangumi_repo_id, f'regular/{select}.zip', repo_type='dataset')


@contextmanager
def make_reg_dir(bangumi_repo_id: Optional[str] = None, bangumi_select: str = 'normal',
                 bangumi_reg_tags: List[str] = None, use_generic_reg: bool = True):
    @contextmanager
    def _yield_default_reg() -> ContextManager[Tuple[str, str]]:
        if use_generic_reg:
            with get_default_reg_dir() as (dir_, cache):
                yield dir_, cache
        else:
            yield None, None

    @contextmanager
    def _yield_bangumi_reg() -> ContextManager[Tuple[str, str]]:
        if bangumi_repo_id:
            with get_bangumi_reg_dir(bangumi_repo_id, bangumi_select) as (dir_, cache):
                yield dir_, cache
        else:
            yield None, None

