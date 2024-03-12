import glob
import json
import logging
import math
import os
import shutil
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image
from hbutils.system import TemporaryDirectory
from hcpdiff.infer_workflow import WorkflowRunner
from hcpdiff.utils import load_config_with_cli
from imgutils.data import load_image
from imgutils.sd import SDMetaData

from ..utils import data_to_cli_args

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

_DEFAULT_INFER_CFG_FILE_LORA = 'cfgs/workflow/anime/highres_fix_anime_lora.yaml'
_DEFAULT_INFER_CFG_FILE_LORA_SIMPLE = 'cfgs/workflow/anime/highres_fix_anime_lora_simple.yaml'
_DEFAULT_INFER_CFG_FILE_LOKR = 'cfgs/workflow/anime/highres_fix_anime_lokr.yaml'
_DEFAULT_INFER_CFG_FILE_LOKR_PIVOTAL = 'cfgs/workflow/anime/highres_fix_anime_lokr_pivotal.yaml'
_DEFAULT_INFER_MODEL = 'Meina/MeinaMix_V11'

_KNOWN_MODEL_HASHES = {
    'AIARTCHAN/anidosmixV2': 'EB49192009',
    'stablediffusionapi/anything-v5': None,
    'stablediffusionapi/cetusmix': 'B42B09FF12',
    'Meina/MeinaMix_V10': 'D967BCAE4A',
    'Meina/MeinaMix_V11': '54EF3E3610',
    'Lykon/DreamShaper': 'C33104F6',
    'digiplay/majicMIX_realistic_v6': 'EBDB94D4',
    'stablediffusionapi/abyssorangemix2nsfw': 'D6992792',
    'AIARTCHAN/expmixLine_v2': 'D91B18D1',
    'Yntec/CuteYuki2': 'FBE372BA',
    'stablediffusionapi/counterfeit-v30': '12047227',
    'jzli/XXMix_9realistic-v4': '5D22F204',
    'stablediffusionapi/flat-2d-animerge': 'F279CF76',
    'redstonehero/cetusmix_v4': '838408E0',
    'Meina/Unreal_V4.1': '0503BFAD',
    'Meina/MeinaHentai_V4': '39C0C3B6',
    'Meina/MeinaPastel_V6': 'DA1D535E',
    'KBlueLeaf/kohaku-v4-rev1.2': '87F9E45D',
    'stablediffusionapi/night-sky-yozora-sty': 'D31F707A',
}


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


def raw_draw_images(workdir: str, prompts: List[str], neg_prompts: List[str], seeds: List[int],
                    model_steps: int = 1000, n_repeats: int = 2, pretrained_model: str = _DEFAULT_INFER_MODEL,
                    firstpass_width: int = 512, firstpass_height: int = 768, width: int = 832, height: int = 1216,
                    cfg_scale: float = 7, infer_steps: int = 30,
                    denoising_strength: float = 0.5, hires_steps: int = 20,
                    model_alpha: float = 0.8, clip_skip: int = 2, sample_method: str = 'DPM++ 2M Karras',
                    cfg_file: str = _DEFAULT_INFER_CFG_FILE_LORA, model_tag: str = 'lora') -> List[Image.Image]:
    meta_file = os.path.join(workdir, 'meta.json')
    with open(meta_file, 'r') as f:
        meta_info = json.load(f)
    train_info = meta_info['train']

    unet_file = os.path.join(workdir, 'ckpts', f'unet-{model_steps}.safetensors')
    logging.info(f'Using unet file {unet_file!r} ...')

    with TemporaryDirectory() as emb_dir, TemporaryDirectory() as output_dir:
        for pt_file in glob.glob(os.path.join(workdir, 'ckpts', f'*-{model_steps}.pt')):
            pt_basename = os.path.splitext(os.path.basename(pt_file))[0]
            pt_name = '-'.join(pt_basename.split('-')[:-1])
            logging.info(f'Loading pt {pt_name!r} from pt file {pt_file!r} ...')
            shutil.copyfile(pt_file, os.path.join(emb_dir, f'{pt_name}.pt'))

        cli_args = data_to_cli_args({
            'bs': 1,
            'seed': seeds,

            'pretrained_model': pretrained_model,
            'prompt': prompts,
            'neg_prompt': neg_prompts,
            'N_repeats': n_repeats,

            'clip_skip': clip_skip - 1,
            'models_dir': os.path.join(workdir, 'ckpts'),
            'emb_dir': emb_dir,

            'infer_args': {
                'init_width': firstpass_width,
                'init_height': firstpass_height,
                'width': width,
                'height': height,
                'guidance_scale': cfg_scale,
                'num_inference_steps': infer_steps,
                'hires_inference_steps': hires_steps,
                'denoising_strength': denoising_strength,
                'scheduler': sample_method_to_config(sample_method),
            },

            'output_dir': output_dir,
            model_tag: {
                'alpha': model_alpha,
                'step': model_steps,
                'unet_': train_info['unet_'] if 'unet_' in train_info else {},
                'text_encoder_': train_info['text_encoder_'] if 'text_encoder_' in train_info else {},
            }
        })
        logging.info(f'Infer based on {cfg_file!r}, with {cli_args!r}')
        cfgs = load_config_with_cli(cfg_file, args_list=cli_args)  # skip --cfg

        runner = WorkflowRunner(cfgs)
        runner.start()

        files = sorted([
            (int(os.path.basename(png_file).split('-')[0]), png_file)
            for png_file in glob.glob(os.path.join(output_dir, '*.png'))
        ])
        images = []
        for _, png_file in files:
            image = load_image(png_file)
            image.load()
            images.append(image)

        return images


@dataclass
class Drawing:
    name: str
    prompt: str
    neg_prompt: str
    seed: int
    firstpass_width: int
    firstpass_height: int
    width: int
    height: int
    cfg_scale: float
    infer_steps: int
    denoising_strength: float
    hires_steps: int
    image: Image.Image
    sample_method: str
    clip_skip: int
    model: str
    model_hash: Optional[str] = None

    @property
    def metadata(self) -> SDMetaData:
        meta_info = {
            'Steps': self.infer_steps,
            'Sampler': self.sample_method,
            'CFG scale': self.cfg_scale,
            'Seed': self.seed,
            'Size': (self.firstpass_width, self.firstpass_height),
            'Denoising strength': self.denoising_strength,
            'Hires resize': (self.width, self.height),
            'Hires steps': self.hires_steps,
            'Hires upscaler': 'Latent (bicubic)',
            'Model': self.model,
            'Clip skip': self.clip_skip,
        }
        if self.model_hash:
            meta_info['Model hash'] = self.model_hash

        return SDMetaData(
            prompt=self.prompt,
            neg_prompt=self.neg_prompt,
            parameters=meta_info,
        )

    def save(self, filename):
        self.image.save(filename, pnginfo=self.metadata.pnginfo)


def draw_images(workdir: str, names: List[str], prompts: List[str], neg_prompts: List[str],
                prompt_reprs: List[str], neg_prompt_reprs: List[str],
                seeds: List[int], model_steps: int = 1000, n_repeats: int = 2,
                pretrained_model: str = _DEFAULT_INFER_MODEL, model_hash: Optional[str] = None,
                firstpass_width: int = 512, firstpass_height: int = 768, width: int = 832, height: int = 1216,
                cfg_scale: float = 7, infer_steps: int = 30,
                denoising_strength: float = 0.5, hires_steps: int = 20,
                model_alpha: float = 0.8, clip_skip: int = 2, sample_method: str = 'DPM++ 2M Karras',
                cfg_file: str = _DEFAULT_INFER_CFG_FILE_LORA, model_tag: str = 'lora') -> List[Drawing]:
    model_hash = model_hash or _KNOWN_MODEL_HASHES.get(pretrained_model) or None
    images: List[Image.Image] = raw_draw_images(
        workdir=workdir,
        prompts=prompts,
        neg_prompts=neg_prompts,
        seeds=seeds,
        model_steps=model_steps,
        n_repeats=n_repeats,
        pretrained_model=pretrained_model,
        firstpass_width=firstpass_width,
        firstpass_height=firstpass_height,
        width=width,
        height=height,
        cfg_scale=cfg_scale,
        infer_steps=infer_steps,
        denoising_strength=denoising_strength,
        hires_steps=hires_steps,
        model_alpha=model_alpha,
        clip_skip=clip_skip,
        sample_method=sample_method,
        cfg_file=cfg_file,
        model_tag=model_tag,
    )

    retval = []
    for name, image, seed, prompt, neg_prompt, prompt_repr, neg_prompt_repr in (
            zip(names, images, seeds, prompts, neg_prompts, prompt_reprs, neg_prompt_reprs)):
        retval.append(Drawing(
            name=name,
            prompt=prompt_repr if prompt_repr is not None else prompt,
            neg_prompt=neg_prompt_repr if neg_prompt_repr is not None else neg_prompt,
            seed=seed,
            firstpass_width=firstpass_width,
            firstpass_height=firstpass_height,
            width=width,
            height=height,
            cfg_scale=cfg_scale,
            infer_steps=infer_steps,
            denoising_strength=denoising_strength,
            hires_steps=hires_steps,
            image=image,
            sample_method=sample_method,
            clip_skip=clip_skip,
            model=pretrained_model,
            model_hash=model_hash,
        ))

    return retval


def list_rtags(workdir: str) -> List[Tuple[int, str, Tuple[str, str], Tuple[str, str], int]]:
    items = []
    for rtag_file in glob.glob(os.path.join(workdir, 'rtags', '*.json')):
        with open(rtag_file, 'r') as f:
            data = json.load(f)
        items.append((
            data['index'], data['name'],
            (data['prompt'], data['neg_prompt']),
            (data['prompt_repr'], data['neg_prompt_repr']),
            data['seed']
        ))
    return sorted(items)


def list_rtag_names(workdir: str) -> List[str]:
    return [name for _, name, _, _, _ in list_rtags(workdir)]


def draw_images_for_workdir(workdir: str, model_steps: int, batch_size: int = 32, n_repeats: int = 2,
                            pretrained_model: str = _DEFAULT_INFER_MODEL, model_hash: Optional[str] = None,
                            firstpass_width: int = 512, firstpass_height: int = 768,
                            width: int = 832, height: int = 1216, cfg_scale: float = 7,
                            infer_steps: int = 30, denoising_strength: float = 0.5, hires_steps: int = 20,
                            model_alpha: float = 0.8, clip_skip: int = 2, sample_method: str = 'DPM++ 2M Karras',
                            cfg_file: str = _DEFAULT_INFER_CFG_FILE_LORA, model_tag: str = 'lora') \
        -> List[Drawing]:
    items = list_rtags(workdir)
    names, prompts, neg_prompts, prompt_reprs, neg_prompt_reprs, seeds = [], [], [], [], [], []
    for _, name, (prompt, neg_prompt), (prompt_repr, neg_prompt_repr), seed in items:
        names.append(name)
        prompts.append(prompt)
        neg_prompts.append(neg_prompt)
        prompt_reprs.append(prompt_repr)
        neg_prompt_reprs.append(neg_prompt_repr)
        seeds.append(seed)

    batch_count = int(math.ceil(len(names) / batch_size))
    retval = []
    for i in range(batch_count):
        while True:
            try:
                new_items = draw_images(
                    workdir=workdir,
                    names=names[i * batch_size: (i + 1) * batch_size],
                    prompts=prompts[i * batch_size: (i + 1) * batch_size],
                    neg_prompts=neg_prompts[i * batch_size: (i + 1) * batch_size],
                    prompt_reprs=prompt_reprs[i * batch_size: (i + 1) * batch_size],
                    neg_prompt_reprs=neg_prompt_reprs[i * batch_size: (i + 1) * batch_size],
                    seeds=seeds[i * batch_size: (i + 1) * batch_size],
                    model_steps=model_steps,
                    model_hash=model_hash,
                    n_repeats=n_repeats,
                    pretrained_model=pretrained_model,
                    firstpass_width=firstpass_width,
                    firstpass_height=firstpass_height,
                    width=width,
                    height=height,
                    cfg_scale=cfg_scale,
                    infer_steps=infer_steps,
                    denoising_strength=denoising_strength,
                    hires_steps=hires_steps,
                    model_alpha=model_alpha,
                    clip_skip=clip_skip,
                    sample_method=sample_method,
                    cfg_file=cfg_file,
                    model_tag=model_tag,
                )
            except RuntimeError as err:
                n_repeats += 1
                warnings.warn(f'Runtime error occurred, n_repeats set to {n_repeats!r}, err: {err!r}')
            else:
                retval.extend(new_items)
                break

    return retval
