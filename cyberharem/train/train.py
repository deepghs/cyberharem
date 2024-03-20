import json
import logging
import os.path
import pathlib
import random
import re
import shutil
import subprocess
from contextlib import contextmanager
from typing import ContextManager, Optional, List, Union

import numpy as np
import toml
from hbutils.design import SingletonMark
from hbutils.system import TemporaryDirectory
from hfutils.operate import download_archive_as_directory
from hfutils.operate.base import RepoTypeTyping, get_hf_fs
from huggingface_hub import hf_hub_download
from imgutils.metrics import ccip_extract_feature
from tqdm.auto import tqdm

from .reg import prepare_reg_dataset, default_reg_dataset, bangumi_reg_dataset
from .tags import save_recommended_tags
from ..utils import get_exec_from_venv, yield_all_images, is_txt_file, file_sha256, dict_merge, \
    NOT_EXIST, IGNORE


@contextmanager
def load_reg_dataset(bangumi_repo_id: Optional[str] = None, bangumi_select: str = 'normal',
                     bangumi_prefix_tag: str = 'anime_style', generic_scale: Optional[int] = None,
                     use_reg: bool = False, latent_cache_id: Optional[str] = None, balance: bool = True) -> Optional[
    str]:
    if not use_reg:
        yield None
    else:
        regs = []
        regs.append(default_reg_dataset(generic_scale))
        if bangumi_repo_id:
            regs.append((bangumi_reg_dataset(bangumi_repo_id, bangumi_select), [bangumi_prefix_tag]))

        with prepare_reg_dataset(*regs, cache_id=latent_cache_id, balance=balance) as dir_:
            yield dir_


def _extract_features_from_directory(dataset_dir):
    features = []
    for file in tqdm(list(yield_all_images(dataset_dir)), desc='Extract Features'):
        features.append(ccip_extract_feature(file))

    return np.stack(features)


@contextmanager
def load_train_dataset(repo_id: str, prefix_tags: List[str] = None,
                       dataset_name: str = 'stage3-p480-1200', revision: str = 'main',
                       repo_type: RepoTypeTyping = 'dataset') -> ContextManager[str]:
    prefix_tags = list(prefix_tags or [])
    logging.info(f'Loading dataset from {repo_id}@{revision}, {dataset_name}, with prefix tags: {prefix_tags!r} ...')
    with TemporaryDirectory() as td:
        subdir_name = '1_1girl'
        ds_dir = os.path.join(td, subdir_name)
        download_archive_as_directory(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            file_in_repo=f'dataset-{dataset_name}.zip',
            local_directory=ds_dir,
        )

        if prefix_tags:
            for root, dirs, files in os.walk(td):
                for file in files:
                    src_file = os.path.join(root, file)
                    if is_txt_file(src_file):
                        origin_prompt = pathlib.Path(src_file).read_text().strip()
                        with open(src_file, 'w') as f:
                            f.write(f'{", ".join(prefix_tags)}, {origin_prompt}')

        yield td


_TRAIN_DIR = os.path.abspath(os.path.dirname(__file__))


def _get_toml_file(file):
    if os.path.exists(os.path.abspath(file)):
        return os.path.abspath(file)
    elif os.path.exists(os.path.abspath(os.path.join(_TRAIN_DIR, file))):
        return os.path.abspath(os.path.join(_TRAIN_DIR, file))
    else:
        raise FileNotFoundError(f'Configuration file {file!r} not found.')


def _load_toml_file(file):
    file = _get_toml_file(file)
    logging.info(f'Loading training config file {file!r} ...')
    return toml.load(file)


@contextmanager
def _use_toml_cfg_file(template_file: str, configs: dict) -> ContextManager[str]:
    with TemporaryDirectory() as td:
        cfg_file = os.path.join(td, 'train_cfg.toml')
        with open(cfg_file, 'w') as f:
            toml.dump(dict_merge(_load_toml_file(template_file), configs), f)
        yield cfg_file


CFG_FILE = SingletonMark('config_file')

_ACCELERATE_EXEC = shutil.which('accelerate')
_KOHYA_TRAIN_TEMPLATE = [_ACCELERATE_EXEC, 'launch', 'train_network.py', '--config_file', CFG_FILE]
_KOHYA_WORKDIR = None


def _set_kohya_command(args: List[Union[str, object]], workdir: str):
    global _KOHYA_TRAIN_TEMPLATE, _KOHYA_WORKDIR
    logging.info(f'Kohya train command has been changed from {_KOHYA_TRAIN_TEMPLATE!r} '
                 f'to {args!r}.')
    _KOHYA_TRAIN_TEMPLATE = args
    _KOHYA_WORKDIR = workdir


def _get_kohya_train_command(cfg_file) -> List[str]:
    return [cfg_file if item is CFG_FILE else str(item) for item in _KOHYA_TRAIN_TEMPLATE]


_CONDA_EXEC = shutil.which('conda')


def set_kohya_from_conda_dir(conda_env_name: str, kohya_directory: str):
    if not _CONDA_EXEC:
        raise EnvironmentError('conda command not found, please install conda and check if it is installed properly.')
    else:
        _set_kohya_command([
            _CONDA_EXEC, 'run', '--live-stream', '-n', conda_env_name,
            'accelerate', 'launch', os.path.join(kohya_directory, 'train_network.py'),
            '--config_file', CFG_FILE,
        ], kohya_directory)


def set_kohya_from_venv_dir(kohya_directory: str):
    _set_kohya_command([
        get_exec_from_venv(os.path.join(kohya_directory, 'venv'), exec_name='accelerate'),
        'launch', os.path.join(kohya_directory, 'train_network.py'),
        '--config_file', CFG_FILE,
    ], kohya_directory)


def _run_kohya_train_command(cfg_file: str):
    if not _KOHYA_WORKDIR:
        raise EnvironmentError('Kohya work directory not assigned. '
                               'Please use `set_kohya_from_conda_dir` or `set_kohya_from_venv_dir` to assign it.')

    commands = _get_kohya_train_command(cfg_file)
    logging.info(f'Running kohya train command with {commands!r}, on workdir {_KOHYA_WORKDIR!r} ...')
    terminal_size = os.get_terminal_size()
    process = subprocess.run(
        commands, cwd=_KOHYA_WORKDIR,
        env={
            **os.environ,
            'COLUMNS': str(terminal_size.columns),
            'LINES': str(terminal_size.lines),
        }
    )
    process.check_returncode()


def train_lora(ds_repo_id: str, dataset_name: str = 'stage3-p480-1200', workdir: Optional[str] = None,
               template_file: str = 'ch_lora_sd15.toml', pretrained_model: str = None,
               seed: int = None, use_reg: bool = True, latent_cache_id: Optional[str] = None,
               bs: int = 8, unet_lr: float = 0.0006, te_lr: float = 0.0006, train_te: bool = False,
               dim: int = 4, alpha: int = 2, resolution: int = 720, res_ratio: float = 2.2,
               comment: str = None):
    hf_fs = get_hf_fs()
    meta = json.loads(hf_fs.read_text(f'datasets/{ds_repo_id}/meta.json'))
    dataset_size = meta['packages'][dataset_name]['size']
    if re.fullmatch(r'^[a-z\\d_]+$', meta['name']):
        name = meta['name']
    else:
        name = ds_repo_id.split('/')[-1]

    workdir = workdir or os.path.join('runs', name)
    os.makedirs(workdir, exist_ok=True)
    save_recommended_tags(name, meta['clusters'], workdir)

    kohya_save_dir = os.path.abspath(os.path.join(workdir, 'kohya'))
    os.makedirs(kohya_save_dir, exist_ok=True)

    pretrained_model = pretrained_model or os.environ.get('CH_TRAIN_BASE_MODEL') or hf_hub_download(
        repo_id='deepghs/animefull-latest-ckpt',
        repo_type='model',
        filename='model.ckpt',
    )
    latent_cache_id = latent_cache_id or file_sha256(pretrained_model)

    with load_train_dataset(repo_id=ds_repo_id, prefix_tags=[name], dataset_name=dataset_name) as train_dir, \
            load_reg_dataset(bangumi_repo_id=meta['bangumi'], use_reg=use_reg,
                             latent_cache_id=latent_cache_id) as reg_dir:
        with _use_toml_cfg_file(template_file, {
            'Basics': {
                'pretrained_model_name_or_path': pretrained_model,
                'train_data_dir': train_dir,
                'reg_data_dir': reg_dir if reg_dir else NOT_EXIST,
                'seed': seed or random.randint(0, (1 << 31) - 1),
                'resolution': f'{resolution},{resolution}'
            },
            'Save': {
                'output_dir': kohya_save_dir,
                'output_name': name,
            },
            'Network_setup': {
                'network_dim': dim,
                'network_alpha': alpha,
                'network_train_unet_only': not bool(train_te),
                'network_train_text_encoder_only': False,
            },
            'Optimizer': {
                'train_batch_size': bs,
                'unet_lr': unet_lr,
                'text_encoder_lr': te_lr,
            },
            'ARB': {
                'min_bucket_reso': int(resolution // res_ratio),
                'max_bucket_reso': int(resolution * res_ratio),
            },
            'Others': {
                'training_comment': comment if comment else IGNORE,
            }
        }) as cfg_file:
            workdir_cfg_file = os.path.abspath(os.path.join(workdir, 'train.toml'))
            shutil.copy(cfg_file, workdir_cfg_file)

            with open(os.path.join(workdir, 'meta.json'), 'w', encoding='utf-8') as f:
                json.dump({
                    'base_model_type': 'SD1.5',
                    'train_type': 'LoRA',
                    'dataset': {
                        'repository': ds_repo_id,
                        'size': dataset_size,
                        'name': dataset_name,
                    },
                    'core_tags': meta['core_tags'],
                    'bangumi': meta['bangumi'],
                    'name': name,
                    'display_name': meta['display_name'],
                    'version': meta['version'],
                }, f, indent=4, sort_keys=True, ensure_ascii=False)

            _run_kohya_train_command(workdir_cfg_file)

    return workdir
