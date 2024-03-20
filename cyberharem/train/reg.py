import logging
import os
import pathlib
import re
import shutil
from contextlib import contextmanager
from typing import Optional, List, Union, Tuple

from filelock import FileLock
from hbutils.reflection import nested_with
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import download_archive_as_directory
from hfutils.operate.base import RepoTypeTyping
from tqdm import tqdm

from ..utils import is_txt_file, is_image_file, is_npz_file


def _reg_root_dir() -> str:
    if not os.environ.get('REG_HOME'):
        raise EnvironmentError('Please assign your REG_HOME environment variable first. '
                               'This directory is used to manage the reg datasets.')
    return os.environ['REG_HOME']


class RegDataset:
    def __init__(self, reg_name):
        self.reg_name = reg_name
        self.root_dir = os.path.join(_reg_root_dir(), self.reg_name)
        self.dataset_dir = os.path.join(self.root_dir, 'dataset')
        self.caches_dir = os.path.join(self.root_dir, 'caches')
        self.lock = FileLock(os.path.join(self.root_dir, '.lock'))
        self._image_cnt = None

    @property
    def image_count(self) -> int:
        if self._image_cnt is None:
            image_count = 0
            for file in os.listdir(self.dataset_dir):
                if is_image_file(file):
                    image_count += 1
            self._image_cnt = image_count
        return self._image_cnt

    @contextmanager
    def mock_to_dir(self, directory, cache_id: Optional[str], prefix_tags: List[str] = None):
        os.makedirs(directory, exist_ok=True)
        prefix_tags = list(prefix_tags or [])
        logging.info('Mocking dataset files ...')
        for file in tqdm(os.listdir(self.dataset_dir)):
            src_file = os.path.join(self.dataset_dir, file)
            dst_file = os.path.join(directory, file)
            if is_txt_file(file):
                if not prefix_tags:
                    shutil.copy(src_file, dst_file)
                else:
                    origin_text = pathlib.Path(src_file).read_text().strip()
                    with open(dst_file, 'w') as f:
                        print(', '.join(prefix_tags), file=f, end='')
                        if origin_text:
                            print(f', {origin_text}', file=f)
            elif is_image_file(file):
                os.symlink(src_file, dst_file)
            else:
                logging.warning(f'Unknown file {file!r} in reg dataset {self.reg_name!r}, skipped.')

        if cache_id is not None:
            logging.info('Loading latent cache files ...')
            cache_dir = os.path.join(self.caches_dir, cache_id)
            os.makedirs(cache_dir, exist_ok=True)

            for file in tqdm(os.listdir(cache_dir)):
                src_file = os.path.join(cache_dir, file)
                dst_file = os.path.join(directory, file)
                if is_npz_file(src_file):
                    os.symlink(src_file, dst_file)
                else:
                    logging.warning(f'Unknown file {file!r} in reg dataset {self.reg_name!r}@{cache_id!r}, skipped.')
        else:
            cache_dir = None

        try:
            yield directory
        finally:
            if cache_dir is not None:
                logging.info('Sync cache files back to reg dataset ...')
                for file in tqdm(os.listdir(directory)):
                    src_file = os.path.join(directory, file)
                    dst_file = os.path.join(cache_dir, file)
                    if not os.path.islink(src_file):
                        if is_npz_file(src_file):
                            shutil.copyfile(src_file, dst_file)
                        elif is_txt_file(src_file):
                            pass  # just do nothing
                        else:
                            logging.warning(
                                f'Unknown file {file!r} in cached reg dataset {self.reg_name!r}@{cache_id!r}, '
                                f'skipped and not synced back.')

    @classmethod
    def initialize_reg_dataset(cls, reg_name: str, repo_id: str, archive_file: str,
                               repo_type: RepoTypeTyping = 'dataset') -> 'RegDataset':
        _reg_dir = os.path.join(_reg_root_dir(), reg_name)
        _reg_ds_dir = os.path.join(_reg_dir, 'dataset')
        _reg_cache_dir = os.path.join(_reg_dir, 'caches')
        os.makedirs(_reg_cache_dir, exist_ok=True)

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
            else:
                logging.info(f'Reg dataset {reg_name!r} already downloaded, skipped.')

        return cls(reg_name)


_DEFAULT_REG = ('deepghs/reg_experiment', 'all_webp.zip')


def default_reg_dataset(scale: Optional[int] = None) -> RegDataset:
    return RegDataset.initialize_reg_dataset(
        reg_name=f'generic_{scale}' if scale else 'generic',
        repo_id='deepghs/reg_experiment',
        archive_file=f'all_webp_{scale}.zip' if scale else 'all_webp.zip',
    )
    pass


def bangumi_reg_dataset(bangumi_repo_id: str, select: str = 'normal') -> RegDataset:
    bangumi_name = bangumi_repo_id.split('/')[-1]
    return RegDataset.initialize_reg_dataset(
        reg_name=f'animereg_{bangumi_name}',
        repo_id=bangumi_repo_id,
        archive_file=f'regular/{select}.zip',
    )


_RegItemTyping = Union[RegDataset, Tuple[RegDataset, List[str]]]


def _name_safe(name_text):
    return re.sub(r'[\W_]+', '_', name_text).strip('_')


@contextmanager
def prepare_reg_dataset(*regs: _RegItemTyping, cache_id: Optional[str] = None, balance: bool = True) -> str:
    with TemporaryDirectory() as td:
        logging.info(f'Preparing reg dataset on {td!r} ...')
        reg_items = []
        max_count = 1
        for reg_item in regs:
            if isinstance(reg_item, tuple):
                reg, prefix_tags = reg_item
            else:
                reg, prefix_tags = reg_item, []
            logging.info(f'Checking reg dataset {reg.reg_name!r} ...')
            image_count = reg.image_count
            reg_items.append((reg, prefix_tags, image_count))
            max_count = max(max_count, image_count)

        mocks = []
        for reg, prefix_tags, image_count in reg_items:
            repeats = (max_count // image_count) if balance else 1
            logging.info(f'Preparing reg dataset {reg.reg_name!r}, with {plural_word(image_count, "image")}, '
                         f'repeat(s): {repeats}, prefix tags: {prefix_tags!r} ...')

            subdir_name = f'{repeats}_{_name_safe(reg.reg_name)}'
            mocks.append(reg.mock_to_dir(os.path.join(td, subdir_name), cache_id, prefix_tags))

        with nested_with(*mocks):
            yield td
