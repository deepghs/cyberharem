import json
import logging
import math
import os.path
import pathlib
import re
from contextlib import contextmanager
from typing import ContextManager, Optional, Tuple

import numpy as np
from hbutils.system import TemporaryDirectory
from hcpdiff.train_ac import Trainer
from hcpdiff.train_ac_single import TrainerSingleCard
from hcpdiff.utils import load_config_with_cli
from hfutils.operate import download_archive_as_directory
from hfutils.operate.base import RepoTypeTyping, get_hf_fs
from imgutils.metrics import ccip_extract_feature
from tqdm.auto import tqdm
from waifuc.utils import get_file_type

from .embedding import create_embedding
from .reg import get_default_reg_dir, get_bangumi_reg_dir
from .tags import save_recommended_tags
from ..utils import data_to_cli_args


@contextmanager
def _get_reg_dir(bangumi_repo_id: Optional[str] = None) -> ContextManager[Tuple[str, str, int]]:
    if not bangumi_repo_id:
        with get_default_reg_dir() as (dir_, cache):
            yield dir_, cache, 20
    else:
        with get_bangumi_reg_dir(bangumi_repo_id) as (dir_, cache):
            yield dir_, cache, 10


def _extract_features_from_directory(dataset_dir):
    features = []
    for file in tqdm(os.listdir(dataset_dir), desc='Extract Features'):
        path = os.path.join(dataset_dir, file)
        type_ = get_file_type(path)
        if type_ and 'image' in type_:
            features.append(ccip_extract_feature(path))

    return np.stack(features)


@contextmanager
def load_dataset_from_repository(repo_id: str, dataset_name: str = 'stage3-p480-800',
                                 revision: str = 'main', repo_type: RepoTypeTyping = 'dataset') -> ContextManager[str]:
    logging.info(f'Loading dataset from {repo_id}@{revision}, {dataset_name} ...')
    with TemporaryDirectory() as dltmp:
        ds_dir = os.path.join(dltmp, 'dataset')
        download_archive_as_directory(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            file_in_repo=f'dataset-{dataset_name}.zip',
            local_directory=ds_dir,
        )
        yield ds_dir


_DEFAULT_TRAIN_MODEL = 'deepghs/animefull-latest'
_DEFAULT_TRAIN_CFG_PLORA = 'cfgs/train/examples/lora_anime_character_reg_v1.5.yaml'


def train_plora(ds_repo_id: str, dataset_name: str = 'stage3-p480-800',
                keep_ckpts: int = 40, bs: int = 4, pretrained_model: str = _DEFAULT_TRAIN_MODEL,
                workdir: str = None, emb_n_words: int = 4, clip_skip: int = 2,
                max_epochs: int = 40, min_epochs: int = 10, min_steps: int = 800, max_steps: int = 10000,
                train_resolution: int = 720, max_reg_bs: int = 16, tag_dropout: float = 0.1,
                pt_lr: float = 0.03, unet_lr: float = 1e-4, unet_rank: float = 0.01,
                single_card: bool = True, force: bool = False) -> str:
    hf_fs = get_hf_fs()
    meta = json.loads(hf_fs.read_text(f'datasets/{ds_repo_id}/meta.json'))

    if dataset_name not in meta['packages']:
        raise ValueError(f'Unknown dataset {dataset_name!r} in repository {ds_repo_id!r}.')
    if meta['packages'][dataset_name]['type'] != "IMG+TXT":
        raise TypeError(f'Dataset {dataset_name!r}\'s type is {meta["packages"][dataset_name]["type"]}, '
                        f'cannot be used for training.')
    dataset_size = meta['packages'][dataset_name]['size']
    train_steps = min(max(dataset_size * max_epochs / bs, min_steps), max(dataset_size * min_epochs / bs, max_steps))
    train_steps = int(math.ceil(train_steps / keep_ckpts) * keep_ckpts)
    save_per_steps = train_steps // keep_ckpts
    reg_bs = min(max(round(max_steps * (bs + 1) / train_steps) - bs, 1), max_reg_bs)

    if re.fullmatch(r'^[a-z\\d_]+$', meta['name']):
        name = meta['name']
    else:
        name = ds_repo_id.split('/')[-1]
    workdir = workdir or os.path.join('runs', name)
    os.makedirs(workdir, exist_ok=True)

    ok_file = os.path.join(workdir, '.train-ok')
    if os.path.exists(ok_file) and not force:
        logging.info(f'Already trained on {workdir!r}, skipped.')
        return workdir

    save_recommended_tags(name, meta['clusters'], workdir)

    with load_dataset_from_repository(ds_repo_id, dataset_name) as dataset_dir, \
            _get_reg_dir(meta['bangumi']) as (reg_dataset_dir, reg_cache, reg_buckets):
        features = _extract_features_from_directory(dataset_dir)
        with open(os.path.join(workdir, 'features.npy'), 'wb') as f:
            np.save(f, features)

        emb_init_text = ', '.join(['1girl', *meta['core_tags']])
        with TemporaryDirectory() as embs_dir:
            logging.info(f'Creating embeddings {name!r} at {embs_dir!r}, '
                         f'n_words: {emb_n_words!r}, init_text: {emb_init_text!r}, '
                         f'pretrained_model: {pretrained_model!r}.')
            create_embedding(
                name, emb_n_words, emb_init_text,
                replace=True,
                pretrained_model=pretrained_model,
                embs_dir=embs_dir,
            )

            train_cfgs = {
                'model': {
                    'pretrained_model_name_or_path': pretrained_model,
                    'clip_skip': clip_skip - 1,
                },
                'train': {
                    'train_steps': train_steps,
                    'save_step': save_per_steps,
                },
                'character_name': name,
                'exp_dir': workdir,
                'dataset': {
                    'dir': dataset_dir,
                    'bs': bs,
                    'resolution': train_resolution,
                    'num_bucket': 5,
                },
                'reg_dataset': {
                    'dir': reg_dataset_dir,
                    'cache': reg_cache,
                    'bs': reg_bs,
                    'resolution': train_resolution,
                    'num_bucket': reg_buckets,
                },
                'tag_dropout': tag_dropout,
                'pt': {
                    'emb_dir': embs_dir,
                    'lr': pt_lr,
                },
                'unet_': {
                    'lr': unet_lr,
                    'rank': unet_rank,
                }
            }
            if reg_bs == 0:
                train_cfgs['data'] = {'dataset_class': '---'}
            with open(os.path.join(workdir, 'meta.json'), 'w', encoding='utf-8') as f:
                json.dump({
                    'base_model_type': 'SD1.5',
                    'train_type': 'Pivotal LoRA',
                    'dataset': {
                        'repository': ds_repo_id,
                        'size': dataset_size,
                        'name': dataset_name,
                    },
                    'train': train_cfgs,
                    'core_tags': meta['core_tags'],
                    'bangumi': meta['bangumi'],
                    'name': name,
                    'display_name': meta['display_name'],
                    'version': meta['version'],
                }, f, indent=4, sort_keys=True, ensure_ascii=False)

            cli_args = data_to_cli_args(train_cfgs)
            conf = load_config_with_cli(_DEFAULT_TRAIN_CFG_PLORA, args_list=cli_args)  # skip --cfg

            logging.info(f'Training with {_DEFAULT_TRAIN_CFG_PLORA!r}, args: {cli_args!r} ...')
            if single_card:
                logging.info('Training with single card ...')
                trainer = TrainerSingleCard(conf)
            else:
                logging.info('Training with non-single cards ...')
                trainer = Trainer(conf)

            trainer.train()

    pathlib.Path(ok_file).touch()
    return workdir


_DEFAULT_TRAIN_CFG_LORA = 'cfgs/train/examples/lora_anime_character_reg_v1.5_simple.yaml'


def train_lora(ds_repo_id: str, dataset_name: str = 'stage3-p480-800',
               keep_ckpts: int = 40, bs: int = 4, pretrained_model: str = _DEFAULT_TRAIN_MODEL,
               workdir: str = None, clip_skip: int = 2,
               max_epochs: int = 40, min_epochs: int = 10, min_steps: int = 800, max_steps: int = 10000,
               train_resolution: int = 720, max_reg_bs: int = 16, tag_dropout: float = 0.1,
               unet_lr: float = 1e-4, unet_rank: int = 8,
               text_encoder_lr: float = 1e-5, text_encoder_rank: int = 4,
               single_card: bool = True, force: bool = False) -> str:
    hf_fs = get_hf_fs()
    meta = json.loads(hf_fs.read_text(f'datasets/{ds_repo_id}/meta.json'))

    if dataset_name not in meta['packages']:
        raise ValueError(f'Unknown dataset {dataset_name!r} in repository {ds_repo_id!r}.')
    if meta['packages'][dataset_name]['type'] != "IMG+TXT":
        raise TypeError(f'Dataset {dataset_name!r}\'s type is {meta["packages"][dataset_name]["type"]}, '
                        f'cannot be used for training.')
    dataset_size = meta['packages'][dataset_name]['size']
    train_steps = min(max(dataset_size * max_epochs / bs, min_steps), max(dataset_size * min_epochs / bs, max_steps))
    train_steps = int(math.ceil(train_steps / keep_ckpts) * keep_ckpts)
    save_per_steps = train_steps // keep_ckpts
    reg_bs = min(max(round(max_steps * (bs + 1) / train_steps) - bs, 1), max_reg_bs)

    if re.fullmatch(r'^[a-z\\d_]+$', meta['name']):
        name = meta['name']
    else:
        name = ds_repo_id.split('/')[-1]
    workdir = workdir or os.path.join('runs', name)
    os.makedirs(workdir, exist_ok=True)

    ok_file = os.path.join(workdir, '.train-ok')
    if os.path.exists(ok_file) and not force:
        logging.info(f'Already trained on {workdir!r}, skipped.')
        return workdir

    save_recommended_tags(name, meta['clusters'], workdir)

    with load_dataset_from_repository(ds_repo_id, dataset_name) as dataset_dir, \
            _get_reg_dir(meta['bangumi']) as (reg_dataset_dir, reg_cache, reg_buckets):
        features = _extract_features_from_directory(dataset_dir)
        with open(os.path.join(workdir, 'features.npy'), 'wb') as f:
            np.save(f, features)

        with TemporaryDirectory() as embs_dir:
            logging.info('No embeddings will be created when training simple lora.')
            train_cfgs = {
                'model': {
                    'pretrained_model_name_or_path': pretrained_model,
                    'clip_skip': clip_skip - 1,
                },
                'train': {
                    'train_steps': train_steps,
                    'save_step': save_per_steps,
                },
                'character_name': name,
                'exp_dir': workdir,
                'dataset': {
                    'dir': dataset_dir,
                    'bs': bs,
                    'resolution': train_resolution,
                    'num_bucket': 5,
                },
                'reg_dataset': {
                    'dir': reg_dataset_dir,
                    'cache': reg_cache,
                    'bs': reg_bs,
                    'resolution': train_resolution,
                    'num_bucket': reg_buckets,
                },
                'tag_dropout': tag_dropout,
                'pt': {
                    'emb_dir': embs_dir,
                },
                'unet_': {
                    'lr': unet_lr,
                    'rank': unet_rank,
                },
                'text_encoder_': {
                    'lr': text_encoder_lr,
                    'rank': text_encoder_rank,
                }
            }
            if reg_bs == 0:
                train_cfgs['data'] = {'dataset_class': '---'}
            with open(os.path.join(workdir, 'meta.json'), 'w', encoding='utf-8') as f:
                json.dump({
                    'base_model_type': 'SD1.5',
                    'train_type': 'LoRA',
                    'dataset': {
                        'repository': ds_repo_id,
                        'size': dataset_size,
                        'name': dataset_name,
                    },
                    'train': train_cfgs,
                    'core_tags': meta['core_tags'],
                    'bangumi': meta['bangumi'],
                    'name': name,
                    'display_name': meta['display_name'],
                    'version': meta['version'],
                }, f, indent=4, sort_keys=True, ensure_ascii=False)

            cli_args = data_to_cli_args(train_cfgs)
            conf = load_config_with_cli(_DEFAULT_TRAIN_CFG_LORA, args_list=cli_args)  # skip --cfg

            logging.info(f'Training with {_DEFAULT_TRAIN_CFG_LORA!r}, args: {cli_args!r} ...')
            if single_card:
                logging.info('Training with single card ...')
                trainer = TrainerSingleCard(conf)
            else:
                logging.info('Training with non-single cards ...')
                trainer = Trainer(conf)

            trainer.train()

    pathlib.Path(ok_file).touch()
    return workdir


_DEFAULT_TRAIN_CFG_LOKR = 'cfgs/train/examples/lokr_anime_character_reg_v1.5.yaml'


def train_lokr(ds_repo_id: str, dataset_name: str = 'stage3-p480-800',
               keep_ckpts: int = 40, bs: int = 4, pretrained_model: str = _DEFAULT_TRAIN_MODEL,
               workdir: str = None, clip_skip: int = 2,
               max_epochs: int = 40, min_epochs: int = 10, min_steps: int = 800, max_steps: int = 10000,
               train_resolution: int = 720, max_reg_bs: int = 16, tag_dropout: float = 0.1,
               lokr_lr: float = 2e-4, lokr_dim: int = 10000, lokr_alpha: int = 0, lokr_factor: int = 8,
               single_card: bool = True, force: bool = False) -> str:
    hf_fs = get_hf_fs()
    meta = json.loads(hf_fs.read_text(f'datasets/{ds_repo_id}/meta.json'))

    if dataset_name not in meta['packages']:
        raise ValueError(f'Unknown dataset {dataset_name!r} in repository {ds_repo_id!r}.')
    if meta['packages'][dataset_name]['type'] != "IMG+TXT":
        raise TypeError(f'Dataset {dataset_name!r}\'s type is {meta["packages"][dataset_name]["type"]}, '
                        f'cannot be used for training.')
    dataset_size = meta['packages'][dataset_name]['size']
    train_steps = min(max(dataset_size * max_epochs / bs, min_steps), max(dataset_size * min_epochs / bs, max_steps))
    train_steps = int(math.ceil(train_steps / keep_ckpts) * keep_ckpts)
    save_per_steps = train_steps // keep_ckpts
    reg_bs = min(max(round(max_steps * (bs + 1) / train_steps) - bs, 1), max_reg_bs)

    if re.fullmatch(r'^[a-z\\d_]+$', meta['name']):
        name = meta['name']
    else:
        name = ds_repo_id.split('/')[-1]
    workdir = workdir or os.path.join('runs', name)
    os.makedirs(workdir, exist_ok=True)

    ok_file = os.path.join(workdir, '.train-ok')
    if os.path.exists(ok_file) and not force:
        logging.info(f'Already trained on {workdir!r}, skipped.')
        return workdir

    save_recommended_tags(name, meta['clusters'], workdir)

    with load_dataset_from_repository(ds_repo_id, dataset_name) as dataset_dir, \
            _get_reg_dir(meta['bangumi']) as (reg_dataset_dir, reg_cache, reg_buckets):
        features = _extract_features_from_directory(dataset_dir)
        with open(os.path.join(workdir, 'features.npy'), 'wb') as f:
            np.save(f, features)

        with TemporaryDirectory() as embs_dir:
            logging.info('No embeddings will be created when training lokr.')
            train_cfgs = {
                'model': {
                    'pretrained_model_name_or_path': pretrained_model,
                    'clip_skip': clip_skip - 1,
                },
                'train': {
                    'lokr': {
                        'lr': lokr_lr,
                        'dim': lokr_dim,
                        'alpha': lokr_alpha,
                        'factor': lokr_factor,
                    },
                    'train_steps': train_steps,
                    'save_step': save_per_steps,
                },
                'character_name': name,
                'exp_dir': workdir,
                'dataset': {
                    'dir': dataset_dir,
                    'bs': bs,
                    'resolution': train_resolution,
                    'num_bucket': 5,
                },
                'reg_dataset': {
                    'dir': reg_dataset_dir,
                    'cache': reg_cache,
                    'bs': reg_bs,
                    'resolution': train_resolution,
                    'num_bucket': reg_buckets,
                },
                'tag_dropout': tag_dropout,
                'pt': {
                    'emb_dir': embs_dir,
                },
            }
            if reg_bs == 0:
                train_cfgs['data'] = {'dataset_class': '---'}
            with open(os.path.join(workdir, 'meta.json'), 'w', encoding='utf-8') as f:
                json.dump({
                    'base_model_type': 'SD1.5',
                    'train_type': 'LoKr',
                    'dataset': {
                        'repository': ds_repo_id,
                        'size': dataset_size,
                        'name': dataset_name,
                    },
                    'train': train_cfgs,
                    'core_tags': meta['core_tags'],
                    'bangumi': meta['bangumi'],
                    'name': name,
                    'display_name': meta['display_name'],
                    'version': meta['version'],
                }, f, indent=4, sort_keys=True, ensure_ascii=False)

            cli_args = data_to_cli_args(train_cfgs)
            conf = load_config_with_cli(_DEFAULT_TRAIN_CFG_LOKR, args_list=cli_args)  # skip --cfg

            logging.info(f'Training with {_DEFAULT_TRAIN_CFG_LOKR!r}, args: {cli_args!r} ...')
            if single_card:
                logging.info('Training with single card ...')
                trainer = TrainerSingleCard(conf)
            else:
                logging.info('Training with non-single cards ...')
                trainer = Trainer(conf)

            trainer.train()

    pathlib.Path(ok_file).touch()
    return workdir
