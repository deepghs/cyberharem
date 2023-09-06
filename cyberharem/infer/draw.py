import glob
import io
import json
import logging
import os
import shutil
from dataclasses import dataclass
from textwrap import dedent
from typing import List, Union, Optional

import yaml
from PIL.PngImagePlugin import PngInfo
from imgutils.detect import detect_censors

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from PIL import Image
from hbutils.system import TemporaryDirectory
from hcpdiff import Visualizer
from hcpdiff.utils import load_config_with_cli

from ..utils import data_to_cli_args

_DEFAULT_INFER_CFG_FILE = 'cfgs/infer/text2img_anime_lora.yaml'
_DEFAULT_INFER_MODEL = 'Meina/MeinaMix_V11'


def sample_method_to_config(method):
    if method == 'DPM++ SDE Karras':
        return {
            '_target_': 'diffusers.DPMSolverSDEScheduler',
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': 'scaled_linear',
            'use_karras_sigmas': True,
        }
    elif method == 'DPM++ 2M Karras':
        return {
            '_target_': 'diffusers.DPMSolverMultistepScheduler',
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'algorithm_type': 'dpmsolver++',
            'beta_schedule': 'scaled_linear',
            'use_karras_sigmas': True
        }
    elif method == 'Euler a':
        return {
            '_target_': 'diffusers.EulerAncestralDiscreteScheduler',
            'beta_start': 0.00085,
            'beta_end': 0.012,
            'beta_schedule': 'scaled_linear',
        }
    else:
        raise ValueError(f'Unknown sample method - {method!r}.')


def draw_images(
        workdir: str, prompts: Union[str, List[str]], neg_prompts: Union[str, List[str]] = None,
        seeds: Union[int, List[str]] = None, emb_name: str = None, save_cfg: bool = True,
        model_steps: int = 1000, n_repeats: int = 2, pretrained_model: str = _DEFAULT_INFER_MODEL,
        width: int = 512, height: int = 768, gscale: float = 8, infer_steps: int = 30,
        lora_alpha: float = 0.85, output_dir: str = 'output', cfg_file: str = _DEFAULT_INFER_CFG_FILE,
        clip_skip: int = 2, sample_method: str = 'DPM++ 2M Karras',
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

            'vae_optimize': {
                'tiling': False,
            },

            'clip_skip': clip_skip - 1,

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
            },

            'new_components': {
                'scheduler': sample_method_to_config(sample_method),
                'vae': {
                    '_target_': 'diffusers.AutoencoderKL.from_pretrained',
                    'pretrained_model_name_or_path': 'deepghs/animefull-latest',  # path to vae model
                    'subfolder': 'vae',
                }
            }
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
    sample_method: str
    clip_skip: int
    model: str
    model_hash: Optional[str] = None

    @property
    def preview_info(self):
        return dedent(f"""
Prompt: {self.prompt}
Neg Prompt: {self.neg_prompt}
Width: {self.width}
Height: {self.height}
Guidance Scale: {self.gscale}
Sample Method: {self.sample_method}
Infer Steps: {self.steps}
Clip Skip: {self.clip_skip}
Seed: {self.seed}
Model: {self.model}
Safe For Work: {"yes" if self.sfw else "no"}
    """).lstrip()

    @property
    def pnginfo_text(self) -> str:
        with io.StringIO() as sf:
            print(self.prompt, file=sf)
            print(f'Negative prompt: {self.neg_prompt}', file=sf)

            if self.model_hash:
                print(f'Steps: {self.steps}, Sampler: {self.sample_method}, '
                      f'CFG scale: {self.gscale}, Seed: {self.seed}, Size: {self.width}x{self.height}, '
                      f'Model hash: {self.model_hash.lower()}, Model: {self.model}, '
                      f'Clip skip: {self.clip_skip}', file=sf)
            else:
                print(f'Steps: {self.steps}, Sampler: {self.sample_method}, '
                      f'CFG scale: {self.gscale}, Seed: {self.seed}, Size: {self.width}x{self.height}, '
                      f'Model: {self.model}, '
                      f'Clip skip: {self.clip_skip}', file=sf)

            return sf.getvalue()

    @property
    def pnginfo(self) -> PngInfo:
        info = PngInfo()
        info.add_text('parameters', self.pnginfo_text)
        return info


_N_MAX_DRAW = 20


def draw_with_workdir(
        workdir: str, emb_name: str = None, save_cfg: bool = True,
        model_steps: int = 1000, n_repeats: int = 2, pretrained_model: str = _DEFAULT_INFER_MODEL,
        width: int = 512, height: int = 768, gscale: float = 8, infer_steps: int = 30,
        lora_alpha: float = 0.85, output_dir: str = None, cfg_file: str = _DEFAULT_INFER_CFG_FILE,
        clip_skip: int = 2, sample_method: str = 'DPM++ 2M Karras', model_hash: Optional[str] = None,
):
    n_pnames, n_prompts, n_neg_prompts, n_seeds, n_sfws = [], [], [], [], []
    for jfile in glob.glob(os.path.join(workdir, 'rtags', '*.json')):
        with open(jfile, 'r', encoding='utf-8') as f:
            data = json.load(f)
            n_pnames.append(data['name'])
            n_prompts.append(data['prompt'])
            n_neg_prompts.append(data['neg_prompt'])
            n_seeds.append(data['seed'])
            n_sfws.append(data['sfw'])

    n_total = len(n_pnames)
    retval = []
    for x in range(0, n_total, _N_MAX_DRAW):
        pnames, prompts, neg_prompts, seeds, sfws = \
            n_pnames[x:x + _N_MAX_DRAW], n_prompts[x:x + _N_MAX_DRAW], n_neg_prompts[x:x + _N_MAX_DRAW], \
                n_seeds[x:x + _N_MAX_DRAW], n_sfws[x:x + _N_MAX_DRAW]

        with TemporaryDirectory() as td:
            _tmp_output_dir = output_dir or td
            draw_images(
                workdir, prompts, neg_prompts, seeds,
                emb_name, save_cfg, model_steps, n_repeats, pretrained_model,
                width, height, gscale, infer_steps, lora_alpha, _tmp_output_dir, cfg_file,
                clip_skip, sample_method,
            )

            for i, (pname, prompt, neg_prompt, seed, sfw) in \
                    enumerate(zip(pnames, prompts, neg_prompts, seeds, sfws), start=1):
                img_file = glob.glob(os.path.join(_tmp_output_dir, f'{i}-*.png'))[0]
                yaml_file = glob.glob(os.path.join(_tmp_output_dir, f'{i}-*.yaml'))[0]
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    seed = yaml.load(f, Loader)['seed']

                img = Image.open(img_file)
                img.load()

                retval.append(Drawing(
                    pname, prompt, neg_prompt, seed,
                    sfw=sfw and len(detect_censors(img, conf_threshold=0.45)) == 0,
                    width=width, height=height, gscale=gscale, steps=infer_steps,
                    image=img, sample_method=sample_method, clip_skip=clip_skip,
                    model=pretrained_model, model_hash=model_hash,
                ))

    return retval
