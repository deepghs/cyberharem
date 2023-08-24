import glob
import logging
import math
import os.path
from typing import Optional, Tuple, Union

from gchar.games.base import Character
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hcpdiff.train_ac import Trainer
from hcpdiff.train_ac_single import TrainerSingleCard
from hcpdiff.utils import load_config_with_cli

from .embedding import create_embedding, _DEFAULT_TRAIN_MODEL
from ..dataset import load_dataset_for_character, save_recommended_tags
from ..utils import data_to_cli_args, get_ch_name

_DEFAULT_TRAIN_CFG = 'cfgs/train/examples/lora_anime_character.yaml'


def train_plora(
        source: Union[str, Character], name: Optional[str] = None,
        epochs: int = 12, save_for_times: int = 15,
        batch_size: int = 4, pretrained_model: str = _DEFAULT_TRAIN_MODEL,
        workdir: str = None, emb_n_words: int = 4, emb_init_text: str = '*0.017',
        unet_rank: float = 8, text_encoder_rank: float = 4,
        cfg_file: str = _DEFAULT_TRAIN_CFG, single_card: bool = True, dataset_size: Tuple[int, int] = (512, 704),
):
    with load_dataset_for_character(source, dataset_size) as (ch, ds_dir):
        if ch is None:
            if name is None:
                raise ValueError(f'Name should be specified when using custom source - {source!r}.')
        else:
            name = name or get_ch_name(ch)

        dataset_size = len(glob.glob(os.path.join(ds_dir, '*.png')))
        logging.info(f'{plural_word(dataset_size, "image")} found in dataset.')

        actual_steps = epochs * dataset_size
        save_per_steps = max(int(math.ceil(actual_steps / save_for_times / 20) * 20), 20)
        steps = int(math.ceil(actual_steps / save_per_steps) * save_per_steps)
        logging.info(f'Training for {plural_word(steps, "step")}, {plural_word(epochs, "epoch")}, '
                     f'save per {plural_word(save_per_steps, "step")} ...')

        workdir = workdir or os.path.join('runs', name)
        os.makedirs(workdir, exist_ok=True)
        save_recommended_tags(ds_dir, name, workdir)

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

            cli_args = data_to_cli_args({
                'train': {
                    'train_steps': steps,
                    'save_step': save_per_steps,
                    'scheduler': {
                        'num_training_steps': steps,
                    }
                },
                'model': {
                    'pretrained_model_name_or_path': pretrained_model,
                },
                'character_name': name,
                'dataset_dir': ds_dir,
                'exp_dir': workdir,
                'unet_rank': unet_rank,
                'text_encoder_rank': text_encoder_rank,
                'tokenizer_pt': {
                    'emb_dir': embs_dir,
                },
                'data': {
                    'dataset1': {
                        'batch_size': batch_size,
                    },
                },
            })
            conf = load_config_with_cli(cfg_file, args_list=cli_args)  # skip --cfg

            logging.info(f'Training with {cfg_file!r}, args: {cli_args!r} ...')
            if single_card:
                logging.info('Training with single card ...')
                trainer = TrainerSingleCard(conf)
            else:
                logging.info('Training with non-single cards ...')
                trainer = Trainer(conf)

            trainer.train()
