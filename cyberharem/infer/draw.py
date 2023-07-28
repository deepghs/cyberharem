import glob
import json
import logging
import os
import shutil
from dataclasses import dataclass
from typing import List, Union

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from PIL import Image
from hbutils.system import TemporaryDirectory
from hcpdiff import Visualizer
from hcpdiff.utils import load_config_with_cli
from imgutils.validate import anime_rating

from ..utils import data_to_cli_args

_DEFAULT_INFER_CFG_FILE = 'cfgs/infer/text2img_anime_lora.yaml'
_DEFAULT_INFER_MODEl = 'stablediffusionapi/anything-v5'


def draw_images(
        workdir: str, prompts: Union[str, List[str]], neg_prompts: Union[str, List[str]] = None,
        seeds: Union[int, List[str]] = None, emb_name: str = None, save_cfg: bool = True,
        model_steps: int = 1000, n_repeats: int = 2, pretrained_model: str = _DEFAULT_INFER_MODEl,
        width: int = 512, height: int = 768, gscale: float = 7.5, infer_steps: int = 30,
        lora_alpha: float = 0.85, output_dir: str = 'output', cfg_file: str = _DEFAULT_INFER_CFG_FILE,
):
    emb_name = emb_name or os.path.basename(workdir)
    with TemporaryDirectory() as emb_dir:
        src_pt_files = glob.glob(os.path.join(workdir, 'ckpts', f'*-{model_steps}.pt'))
        if not src_pt_files:
            raise FileNotFoundError(f'Embedding not found for step {model_steps}.')

        src_pt_file = src_pt_files[0]
        shutil.copyfile(src_pt_file, os.path.join(emb_dir, f'{emb_name}.pt'))

        cli_args = data_to_cli_args({
            'pretrained_model': pretrained_model,
            'N_repeats': n_repeats,

            'bs': 1,
            'num': 1,

            'infer_args': {
                'width': width,
                'height': height,
                'guidance_scale': gscale,
                'num_inference_steps': infer_steps,
            },

            'exp_dir': workdir,
            'model_steps': model_steps,
            'emb_dir': emb_dir,
            'output_dir': output_dir,

            'merge': {
                'alpha': lora_alpha,
                'group1': {

                },
            },
        })
        logging.info(f'Infer based on {cfg_file!r}, with {cli_args!r}')
        cfgs = load_config_with_cli(cfg_file, args_list=cli_args)  # skip --cfg

        N = None
        if isinstance(prompts, list):
            N = len(prompts)
        if isinstance(neg_prompts, list):
            if N is not None and len(neg_prompts) != N:
                raise ValueError(f'Number of prompts ({len(prompts)}) and neg_prompts ({len(neg_prompts)}) not match.')
            N = len(neg_prompts)
        if isinstance(seeds, list):
            if N is not None and len(seeds) != N:
                raise ValueError(f'Number of both prompts ({N}) and seed ({len(seeds)}) not match.')
            N = len(seeds)

        if N is None:
            N = 1
        if not isinstance(prompts, list):
            prompts = [prompts] * N
        if not isinstance(neg_prompts, list):
            neg_prompts = [neg_prompts] * N
        if not isinstance(seeds, list):
            seeds = [seeds] * N

        viser = Visualizer(cfgs)
        viser.vis_to_dir(prompt=prompts, negative_prompt=neg_prompts, seeds=seeds,
                         save_cfg=save_cfg, **cfgs.infer_args)


@dataclass
class Drawing:
    name: str
    prompt: str
    neg_prompt: str
    seed: int
    sfw: bool
    width: int
    height: int
    gscale: float
    steps: int
    image: Image.Image


def draw_with_workdir(
        workdir: str, emb_name: str = None, save_cfg: bool = True,
        model_steps: int = 1000, n_repeats: int = 2, pretrained_model: str = _DEFAULT_INFER_MODEl,
        width: int = 512, height: int = 768, gscale: float = 7.5, infer_steps: int = 30,
        lora_alpha: float = 0.85, output_dir: str = None, cfg_file: str = _DEFAULT_INFER_CFG_FILE,
):
    pnames, prompts, neg_prompts, seeds, sfws = [], [], [], [], []
    for jfile in glob.glob(os.path.join(workdir, 'rtags', '*.json')):
        with open(jfile, 'r', encoding='utf-8') as f:
            data = json.load(f)
            pnames.append(data['name'])
            prompts.append(data['prompt'])
            neg_prompts.append(data['neg_prompt'])
            seeds.append(data['seed'])
            sfws.append(data['sfw'])

    with TemporaryDirectory() as td:
        output_dir = output_dir or td
        draw_images(
            workdir, prompts, neg_prompts, seeds,
            emb_name, save_cfg, model_steps, n_repeats, pretrained_model,
            width, height, gscale, infer_steps, lora_alpha, output_dir, cfg_file
        )

        retval = []
        for i, (pname, prompt, neg_prompt, seed, sfw) in \
                enumerate(zip(pnames, prompts, neg_prompts, seeds, sfws), start=1):
            img_file = glob.glob(os.path.join(output_dir, f'{i}-*.png'))[0]
            yaml_file = glob.glob(os.path.join(output_dir, f'{i}-*.yaml'))[0]
            with open(yaml_file, 'r', encoding='utf-8') as f:
                seed = yaml.load(f, Loader)['seed']

            img = Image.open(img_file)
            img.load()

            retval.append(Drawing(
                pname, prompt, neg_prompt, seed,
                sfw=sfw and anime_rating(img)[0] != 'r18',
                width=width, height=height, gscale=gscale, steps=infer_steps,
                image=img
            ))

        return retval
