import json
import json
import logging
import math
import os.path
import pathlib
import random
import re
import shutil
import subprocess
from contextlib import contextmanager
from typing import ContextManager, Optional, List, Union, Dict

import numpy as np
import toml
from hbutils.design import SingletonMark
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate.base import get_hf_fs
from huggingface_hub import hf_hub_download
from imgutils.metrics import ccip_extract_feature
from tqdm.auto import tqdm

from .reg import prepare_reg_dataset, default_reg_dataset, bangumi_reg_dataset
from .tags import save_recommended_tags
from ..utils import get_exec_from_venv, yield_all_images, is_txt_file, file_sha256, dict_merge, \
    NOT_EXIST, IGNORE, is_image_file


@contextmanager
def load_reg_dataset(bangumi_repo_id: Optional[str] = None, bangumi_select: str = 'normal',
                     bangumi_prefix_tag: str = 'anime_style', generic_scale: Optional[int] = None,
                     use_reg: bool = False, latent_cache_id: Optional[str] = None, balance: bool = True) \
        -> Optional[str]:
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


_DEFAULT_PLS = {
    'head': 1.7,
}


@contextmanager
def load_train_dataset(repo_id: str, prefix_tags: List[str] = None,
                       dataset_name: str = 'stage3-p480-1200', revision: str = 'main',
                       attach_revisions: Optional[List[str]] = None,
                       mls: Optional[Dict[str, float]] = None) -> ContextManager[str]:
    from .dataset import multi_dataset_from_repo
    with multi_dataset_from_repo(repo_id, prefix_tags, dataset_name, revision, attach_revisions, mls) as td:
        yield td


def _gender_predict(train_dir):
    total, boys, girls = 0, 0, 0
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            src_file = os.path.join(root, file)
            if is_txt_file(src_file):
                origin_prompt = pathlib.Path(src_file).read_text().strip()
                tags = re.split(r'\s*,\s*', origin_prompt)
                is_girl = '1girl' in tags
                is_boy = '1boy' in tags
                total += 1
                if is_girl and not is_boy:
                    girls += 1
                elif not is_girl and is_boy:
                    boys += 1

    return boys / total, girls / total


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
def _use_toml_cfg_file(template_file: str, *configs: dict) -> ContextManager[str]:
    with TemporaryDirectory() as td:
        cfg_file = os.path.join(td, 'train_cfg.toml')
        data = _load_toml_file(template_file)
        for config in configs:
            data = dict_merge(data, config)
        with open(cfg_file, 'w') as f:
            toml.dump(data, f)
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


def set_kohya_from_conda_dir(conda_env_name: str, kohya_directory: str,
                             accelerate_extra_args: Optional[List[str]] = None,
                             live_stream: bool = True, no_capture_output: bool = True):
    if not _CONDA_EXEC:
        raise EnvironmentError('conda command not found, please install conda and check if it is installed properly.')
    else:
        _set_kohya_command([
            _CONDA_EXEC, 'run', *(['--live-stream'] if live_stream else []),
            '-n', conda_env_name, *(['--no-capture-output', ] if no_capture_output else []),
            'accelerate', 'launch', *(accelerate_extra_args or []),
            os.path.join(kohya_directory, 'train_network.py'), '--config_file', CFG_FILE,
        ], kohya_directory)


def set_kohya_from_venv_dir(kohya_directory: str, venv_name: str = 'venv'):
    _set_kohya_command([
        get_exec_from_venv(os.path.join(kohya_directory, venv_name), exec_name='accelerate'),
        'launch', os.path.join(kohya_directory, 'train_network.py'),
        '--config_file', CFG_FILE,
    ], kohya_directory)


def _auto_set_kohya_from_env():
    kohya_dir = os.environ.get('CH_KOHYA_DIR')
    if kohya_dir:
        kohya_conda_env = os.environ.get('CH_KOHYA_CONDA_ENV')
        if kohya_conda_env:
            set_kohya_from_conda_dir(
                conda_env_name=kohya_conda_env,
                kohya_directory=kohya_dir,
            )
        else:
            kohya_venv = os.environ.get('CH_KOHYA_VENV')
            if kohya_venv:
                set_kohya_from_venv_dir(
                    kohya_directory=kohya_dir,
                    venv_name=kohya_venv,
                )
            else:
                logging.warning(f'Kohya directory {kohya_dir!r} detected, but cannot determine it is conda or venv. '
                                f'Please explicitly set the environment variable CH_KOHYA_CONDA_ENV or CH_KOHYA_VENV.')
    else:
        logging.info('No local kohya settings found.')


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
        },
        bufsize=0,
    )
    process.check_returncode()


_TRAIN_SET_RANGES = {
    (10000, +math.inf): (6, 1),
    (8000, 10000): (7, 1),
    (6000, 8000): (8, 1),
    (4000, 6000): (10, 1),
    (2000, 4000): (15, 1),
    (1000, 2000): (20, 1),
    (480, 1000): (30, 2),
    (300, 480): (40, 2),
    (100, 300): (60, 3),
    (0, 100): (80, 3),
}


def piecewise_ep(train_set_size: int):
    """
    Created on Wed Mar 20 16:23:32 2024

    @author: rezer
    """
    for range_, return_value in _TRAIN_SET_RANGES.items():
        if range_[0] < train_set_size <= range_[1]:
            return return_value
    else:
        raise ValueError(f'Invalid train set size: {train_set_size!r}')


def count_images_from_train_dir(train_dir) -> int:
    cnt = 0
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            src_file = os.path.join(root, file)
            if is_image_file(src_file):
                cnt += 1

    return cnt


def _auto_init():
    if not _KOHYA_WORKDIR:
        _auto_set_kohya_from_env()


TRAIN_MARK = 'v1.6-alpha0'


def train_lora(ds_repo_id: str, dataset_name: Optional[str] = None, workdir: Optional[str] = None,
               template_file: str = 'ch_lora_sd15.toml', pretrained_model: str = None,
               seed: int = None, use_reg: Optional[bool] = False, latent_cache_id: Optional[str] = None,
               bs: int = 8, unet_lr: float = 0.0006, te_lr: float = 0.0006, train_te: bool = False,
               dim: Optional[int] = None, alpha: int = 2, resolution: int = 720, res_ratio: float = 2.2,
               bangumi_style_tag: str = 'anime_style', comment: str = None, force_retrain: bool = False,
               eps: Optional[int] = None, save_interval: Optional[int] = None,
               ds_attach_revisions: Optional[List[str]] = None, ds_mls: Optional[Dict[str, float]] = None):
    _auto_init()
    hf_fs = get_hf_fs()
    meta = json.loads(hf_fs.read_text(f'datasets/{ds_repo_id}/meta.json'))
    if dataset_name is None:
        if 'stage3-p480-1200' in meta['packages']:
            dataset_name = 'stage3-p480-1200'
        else:
            dataset_name = 'stage3-p180-1200'
    dataset_size = meta['packages'][dataset_name]['size']
    if re.fullmatch(r'^[a-z\\d_]+$', meta['name']):
        name = meta['name']
    else:
        name = ds_repo_id.split('/')[-1]

    workdir = workdir or os.path.join('runs', name)
    os.makedirs(workdir, exist_ok=True)
    last_attempt_file = os.path.join(workdir, 'last_attempt.json')
    trained_flag_file = os.path.join(workdir, '.trained')
    if os.path.exists(trained_flag_file) and not force_retrain:
        logging.info('Model already trained, skipped.')
        return workdir

    kohya_save_dir = os.path.abspath(os.path.join(workdir, 'kohya'))
    os.makedirs(kohya_save_dir, exist_ok=True)

    pretrained_model = pretrained_model or os.environ.get('CH_TRAIN_BASE_MODEL') or hf_hub_download(
        repo_id='deepghs/animefull-latest-ckpt',
        repo_type='model',
        filename='model.ckpt',
    )
    latent_cache_id = latent_cache_id or file_sha256(pretrained_model)

    train_prefix_tags = [name] if not meta['bangumi'] else [name, bangumi_style_tag]
    with load_train_dataset(
            repo_id=ds_repo_id,
            prefix_tags=train_prefix_tags,
            dataset_name=dataset_name,
            attach_revisions=ds_attach_revisions,
            mls=ds_mls,
    ) as train_dir:
        image_count = count_images_from_train_dir(train_dir)
        if use_reg is None:
            if image_count >= 300:
                use_reg = False
            else:
                use_reg = True

        with load_reg_dataset(bangumi_repo_id=meta['bangumi'], bangumi_prefix_tag=bangumi_style_tag,
                              use_reg=use_reg, latent_cache_id=latent_cache_id) as reg_dir:
            features_path = os.path.join(workdir, 'features.npy')
            if os.path.exists(features_path):
                logging.info('Features already exist, use the existing one.')
            else:
                logging.info(f'Extracting features from {train_dir!r}, and saving that to {features_path!r} ...')
                np.save(features_path, _extract_features_from_directory(train_dir))

            r_boy, r_girl = _gender_predict(train_dir)
            if r_boy >= 0.7 or r_boy - r_girl >= 0.25:
                gender = 'boy'
            elif r_girl >= 0.7 or r_girl - r_boy >= 0.25:
                gender = 'girl'
            else:
                gender = 'not_sure'
            logging.info(f'Boy ratio: {r_boy:.3f}, girl ratio: {r_girl:.3f}, gender: {gender!r}.')
            save_recommended_tags(name, meta['clusters'], workdir, gender=gender)

            if eps is None:
                eps, save_interval = piecewise_ep(image_count)
            elif save_interval is None:
                save_interval = max(int(round(eps / 20)), 1)
            logging.info(f'{plural_word(image_count, "word")} detected in training dataset, '
                         f'recommended epochs: {eps}, save interval: {save_interval}.')
            _last_toml_file, _last_lora_file = None, None
            if not dim and os.path.exists(last_attempt_file):
                with open(last_attempt_file) as f:
                    last_attempt = json.load(f)
                last_attempt_workdir = os.path.join(workdir, last_attempt['rel_workdir'])

                retry_reason = last_attempt['info']['reason']
                if retry_reason == 'step_too_low':
                    from ..infer import find_steps_in_workdir
                    last_record = find_steps_in_workdir(last_attempt_workdir).to_dict('records')[-1]
                    _last_toml_file = os.path.join(last_attempt_workdir, 'train.toml')
                    _last_lora_file = last_record['file']
                    logging.info(f'Last attempt found as {last_attempt_workdir!r}, '
                                 f'steps: {last_record["step"]!r}, epoch: {last_record["epoch"]!r}, '
                                 f'lora file: {_last_lora_file!r}, go on training')

                else:
                    logging.warning(f'Unknown retry reason: {retry_reason!r}, ignored.')

            if not dim:
                if image_count <= 100:
                    dim = 4
                elif image_count <= 400:
                    dim = 6
                else:
                    dim = 8
                logging.info(f'Auto selected dim: {dim!r}.')

            seed = seed or random.randint(0, (1 << 30) - 1)
            with _use_toml_cfg_file(_last_toml_file or template_file, {
                'Basics': {
                    'pretrained_model_name_or_path': pretrained_model,
                    'train_data_dir': train_dir,
                    'reg_data_dir': reg_dir if reg_dir else NOT_EXIST,
                    'seed': seed,
                    'resolution': f'{resolution},{resolution}',
                    'max_train_steps': (1 << 31 - 1),
                    'max_train_epochs': eps,
                },
                'Save': {
                    'output_dir': kohya_save_dir,
                    'output_name': name,
                    'save_every_n_epochs': save_interval,
                    'save_every_n_steps': (1 << 31 - 1),
                },
                'Network_setup': {
                    'network_dim': dim,
                    'network_alpha': alpha,
                    'network_train_unet_only': not bool(train_te),
                    'network_train_text_encoder_only': False,
                    'network_weights': _last_lora_file if _last_toml_file else NOT_EXIST,
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
                            'version': meta['version'],
                            'image_count': image_count,
                        },
                        'gender': {
                            'boy': r_boy,
                            'girl': r_girl,
                            'predict': gender,
                        },
                        'core_tags': meta['core_tags'],
                        'bangumi': meta['bangumi'],
                        'name': name,
                        'bangumi_style_name': bangumi_style_tag if meta['bangumi'] else None,
                        'display_name': meta['display_name'],
                        'version': TRAIN_MARK,
                    }, f, indent=4, sort_keys=True, ensure_ascii=False)

                _run_kohya_train_command(workdir_cfg_file)

    pathlib.Path(trained_flag_file).touch()
    return workdir
