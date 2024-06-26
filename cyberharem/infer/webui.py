import glob
import json
import logging
import os.path
import pathlib
import random
import re
from functools import lru_cache
from typing import Optional, List, Set

import pandas as pd
import toml
from hbutils.random import random_sha1_with_timestamp
from hbutils.string import singular_form, plural_word
from hbutils.system import urlsplit
from imgutils.sd import parse_sdmeta_from_text
from imgutils.tagging import remove_underline
from tqdm import tqdm
from webuiapi import WebUIApi, ADetailer

from .steps import find_steps_in_workdir
from .tags import find_tags_from_workdir

_WEBUI_CLIENT: Optional[WebUIApi] = None


def set_webui_server(host="127.0.0.1", port=7860, baseurl=None, use_https=False, **kwargs):
    global _WEBUI_CLIENT
    logging.info(f'Set webui server {"https" if use_https else "http"}://{host}:{port}/{baseurl or ""}')
    _WEBUI_CLIENT = WebUIApi(
        host=host,
        port=port,
        baseurl=baseurl,
        use_https=use_https,
        **kwargs
    )
    _get_client_scripts.cache_clear()


@lru_cache()
def _get_client_scripts() -> Set[str]:
    client = _get_webui_client()
    scripts = client.get_scripts()['txt2img']
    return set(map(str.lower, scripts))


def _has_adetailer() -> bool:
    return 'adetailer' in _get_client_scripts()


def _get_dynamic_prompts_name() -> Optional[str]:
    for name in _get_client_scripts():
        if 'dynamic' in name and 'prompts' in name:
            logging.info(f'Dynamic prompts found, name: {name!r}')
            return name
    logging.error('Dynamic prompts not found.')
    return None


def _set_webui_server_from_env():
    webui_server = os.environ.get('CH_WEBUI_SERVER')
    if webui_server:
        url = urlsplit(webui_server)
        if ':' in url.host:
            host, port = url.host.split(':', maxsplit=1)
            port = int(port)
        else:
            host, port = url.host, 80

        set_webui_server(
            host=host,
            port=port,
            use_https=url.scheme == 'https',
        )
    else:
        logging.info('No webui server settings found.')


def _get_webui_client() -> WebUIApi:
    if _WEBUI_CLIENT:
        return _WEBUI_CLIENT
    else:
        raise OSError('Webui server not set, please set that with `set_webui_server` function.')


class LoraMock:
    def mock_lora(self, local_lora_file: str) -> str:
        raise NotImplementedError

    def unmock_lora(self, lora_name: str):
        raise NotImplementedError


class LocalLoraMock(LoraMock):
    def __init__(self, sd_webui_dir: str):
        self.sd_webui_dir = sd_webui_dir
        self.lora_dir = os.path.abspath(os.path.join(self.sd_webui_dir, 'models', 'Lora', 'automation'))
        os.makedirs(self.lora_dir, exist_ok=True)

    def mock_lora(self, local_lora_file: str) -> str:
        random_sha = random_sha1_with_timestamp()
        _, ext = os.path.splitext(local_lora_file)
        src_file = os.path.abspath(local_lora_file)
        dst_file = os.path.join(self.lora_dir, f'{random_sha}{ext}')
        logging.info(f'Mocking lora file from {src_file!r} to {dst_file!r} ...')
        os.symlink(src_file, dst_file)
        return random_sha

    def unmock_lora(self, lora_name: str):
        files = glob.glob(os.path.join(self.lora_dir, f'{lora_name}.*'))
        if files:
            file = files[0]
            if os.path.islink(file):
                logging.info(f'Unmocking lora {lora_name!r} from {file!r} ...')
                os.unlink(file)
            else:
                raise RuntimeError(f'Mocked lora file {file!r} is not a sym link, cannot unmock.')
        else:
            raise RuntimeError(f'No mocked lora file {lora_name!r} found.')


_WEBUI_LORA_MOCK: Optional[LoraMock] = None


def set_webui_local_dir(webui_local_dir: str):
    lora_dir = os.path.join(webui_local_dir, 'models', 'Lora')
    if os.path.exists(lora_dir):
        global _WEBUI_LORA_MOCK
        logging.info(f'Setting webui local directory {webui_local_dir!r} ...')
        _WEBUI_LORA_MOCK = LocalLoraMock(webui_local_dir)
    else:
        raise EnvironmentError(f'Lora directory not found in {webui_local_dir!r}, '
                               f'please make sure this directory is a valid webui directory.')


def _set_webui_local_dir_with_env():
    webui_dir = os.environ.get('CH_WEBUI_DIR')
    if webui_dir:
        set_webui_local_dir(webui_dir)
    else:
        logging.info('No webui directory settings found.')


def _get_webui_lora_mock() -> LoraMock:
    return _WEBUI_LORA_MOCK


def _auto_init():
    if not _WEBUI_LORA_MOCK:
        _set_webui_local_dir_with_env()
    if not _WEBUI_CLIENT:
        _set_webui_server_from_env()


def infer_with_lora(
        lora_file: str, eye_tags: List[str], df_tags: pd.DataFrame, seed: int,
        batch_size=64, sampler_name='DPM++ 2M Karras', cfg_scale=7, steps=30,
        firstphase_width=512, firstphase_height=768,
        enable_hr: bool = True, hr_resize_x=832, hr_resize_y=1216,
        denoising_strength=0.6, hr_second_pass_steps=20, hr_upscaler='R-ESRGAN 4x+ Anime6B',
        clip_skip: int = 2, lora_alpha: float = 0.8, enable_adetailer: bool = True,
        base_model: str = 'meinamix_v11', extra_tags: Optional[List[str]] = None,
):
    _auto_init()
    mock = _get_webui_lora_mock()
    client = _get_webui_client()
    logging.info(f'Set base model {base_model} ...')
    client.util_set_model(base_model)
    lora_name = mock.mock_lora(lora_file)
    extra_tags = list(extra_tags or [])
    try:
        logging.info(f'Preparing to infer {plural_word(len(df_tags), "image")} ...')
        suffix = f'<lora:{lora_name}:{lora_alpha:.2f}>'
        prompts = []
        names = []
        for i, tag_item in enumerate(df_tags.to_dict('records')):
            prompt = tag_item['prompt'].replace('{', '').replace('}', '').replace('|', '')
            prompts.append(f'{prompt}, {", ".join([*extra_tags, str(i)])}')
            names.append(tag_item['name'])

        full_prompt = f'{{{"|".join(prompts)}}} {suffix}'
        scripts = {}
        dynamic_prompt_name = _get_dynamic_prompts_name()
        if dynamic_prompt_name:
            scripts[dynamic_prompt_name] = {
                "args": [
                    True,
                    True,
                ]
            }
        else:
            raise OSError('No dynamic prompt detected in webui, please install it!')

        if _has_adetailer():
            if enable_adetailer:
                adetailers = [
                    ADetailer(
                        ad_model='face_yolov8n.pt',
                        # ad_model='mediapipe_face_mesh_eyes_only',
                        ad_prompt=f'best eyes, [PROMPT], {", ".join(map(remove_underline, eye_tags))}, '
                                  f'beautiful eyes, extremely detailed eyes, shiny eyes, lively eyes, livid eyes',
                        ad_denoising_strength=denoising_strength,
                        ad_clip_skip=clip_skip,
                    ),
                    ADetailer(ad_model='None'),
                ]
            else:
                logging.info('ADetailer disabled.')
                adetailers = []
        else:
            logging.warning('No Adetailer detected in webui, adetailer will be disabled.')
            adetailers = []

        result = client.txt2img(
            prompt=full_prompt,
            negative_prompt='(worst quality, low quality:1.40), (zombie, sketch, interlocked fingers, comic:1.10), (reference:1.10), lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, white border, (english text, chinese text:1.05), (censored, mosaic censoring, bar censor:1.20)',
            alwayson_scripts=scripts,
            batch_size=batch_size,
            sampler_name=sampler_name,
            cfg_scale=cfg_scale,
            steps=steps,
            firstphase_width=firstphase_width,
            firstphase_height=firstphase_height,
            hr_resize_x=hr_resize_x,
            hr_resize_y=hr_resize_y,
            denoising_strength=denoising_strength,
            hr_second_pass_steps=hr_second_pass_steps,
            hr_upscaler=hr_upscaler,
            seed=seed,
            enable_hr=enable_hr,
            override_settings={
                'CLIP_stop_at_last_layers': clip_skip,
            },
            adetailer=adetailers,
        )
        return list(zip(names, result.images)), lora_name

    finally:
        mock.unmock_lora(lora_name)


def infer_with_workdir(
        workdir: str,
        batch_size=64, sampler_name='DPM++ 2M Karras', cfg_scale=7, steps=30,
        firstphase_width=512, firstphase_height=768,
        enable_hr: bool = True, hr_resize_x=832, hr_resize_y=1216,
        denoising_strength=0.6, hr_second_pass_steps=20, hr_upscaler='R-ESRGAN 4x+ Anime6B',
        clip_skip: int = 2, lora_alpha: float = 0.8, enable_adetailer: bool = True,
        base_model: str = 'meinamix_v11',
):
    _auto_init()

    df_steps = find_steps_in_workdir(workdir)
    logging.info(f'Available steps: {len(df_steps)}\n'
                 f'{df_steps}')

    df_tags = find_tags_from_workdir(workdir)
    logging.info(f'Available prompts: {len(df_tags)}\n'
                 f'{df_tags}')

    with open(os.path.join(workdir, 'meta.json')) as f:
        meta = json.load(f)
    trigger_name = meta['name']
    bangumi_style_tag = meta.get('bangumi_style_name')
    core_tags = meta['core_tags']
    eye_tags = []
    for tag in core_tags:
        is_eye_tag = False
        for word in re.split(r'[\W_]+', tag):
            if word:
                if singular_form(word) == 'eye':
                    is_eye_tag = True
                    break
            else:
                continue
        if is_eye_tag:
            eye_tags.append(tag)

    seed = toml.load(os.path.join(workdir, 'train.toml'))['Basics']['seed']
    for step_item in tqdm(df_steps.to_dict('records')):
        step = step_item['step']
        eval_dir = os.path.join(step_item['workdir'], 'eval')
        os.makedirs(eval_dir, exist_ok=True)
        step_eval_dir = os.path.join(eval_dir, str(step))
        step_eval_infer_okay_file = os.path.join(step_eval_dir, '.inferred')
        if os.path.exists(step_eval_infer_okay_file):
            logging.info(f'Step {step} already inferred, skipped.')
        else:
            os.makedirs(step_eval_dir, exist_ok=True)
            logging.info(f'Infer for step {step} ...')
            pairs, lora_name = infer_with_lora(
                lora_file=step_item['file'],
                eye_tags=eye_tags,
                df_tags=df_tags,
                seed=seed,
                batch_size=batch_size,
                sampler_name=sampler_name,
                cfg_scale=cfg_scale,
                steps=steps,
                firstphase_width=firstphase_width,
                firstphase_height=firstphase_height,
                enable_hr=enable_hr,
                hr_resize_x=hr_resize_x,
                hr_resize_y=hr_resize_y,
                denoising_strength=denoising_strength,
                hr_second_pass_steps=hr_second_pass_steps,
                hr_upscaler=hr_upscaler,
                clip_skip=clip_skip,
                lora_alpha=lora_alpha,
                enable_adetailer=enable_adetailer,
                base_model=base_model,
                extra_tags=[] if not bangumi_style_tag else [bangumi_style_tag],
            )
            for name, image in tqdm(pairs, desc='Save Images'):
                param_text = image.info.get('parameters').replace(lora_name, trigger_name)
                sdmeta = parse_sdmeta_from_text(param_text)
                dst_image_file = os.path.join(step_eval_dir, f'{name}.png')
                image.save(dst_image_file, pnginfo=sdmeta.pnginfo)

            pathlib.Path(step_eval_infer_okay_file).touch()


def infer_for_scale(
        workdir: str,
        batch_size=64, sampler_name='DPM++ 2M Karras', cfg_scale=7, steps=30,
        firstphase_width=512, firstphase_height=768,
        enable_hr: bool = True, hr_resize_x=832, hr_resize_y=1216,
        denoising_strength=0.6, hr_second_pass_steps=20, hr_upscaler='R-ESRGAN 4x+ Anime6B',
        clip_skip: int = 2, lora_alpha: float = 0.8, enable_adetailer: bool = True,
        base_model: str = 'nai', eval_cfgs: Optional[dict] = None,
        max_n_steps: Optional[int] = None, infer_seed_count: int = 5,
        ccip_distance_mode: bool = False
):
    _auto_init()

    from ..eval import eval_for_workdir
    logging.info('Starting evaluation before deployment ...')
    eval_for_workdir(workdir, ccip_distance_mode=ccip_distance_mode, **(eval_cfgs or {}))

    eval_dir = os.path.join(workdir, 'eval')
    df_selected_file = os.path.join(eval_dir, 'metrics_selected.csv')
    df_selected_steps = pd.read_csv(df_selected_file)
    if max_n_steps:
        df_selected_steps = df_selected_steps[:max_n_steps]
    logging.info(f'Steps to infer: {len(df_selected_steps)}\n'
                 f'{df_selected_steps}')

    df_steps = find_steps_in_workdir(workdir)
    lora_files = {item['step']: item['file'] for item in df_steps.to_dict('records')}

    df_tags = find_tags_from_workdir(workdir)
    logging.info(f'Available prompts: {len(df_tags)}\n'
                 f'{df_tags}')

    with open(os.path.join(workdir, 'meta.json')) as f:
        meta = json.load(f)
    trigger_name = meta['name']
    bangumi_style_tag = meta.get('bangumi_style_name')
    core_tags = meta['core_tags']
    eye_tags = []
    for tag in core_tags:
        is_eye_tag = False
        for word in re.split(r'[\W_]+', tag):
            if word:
                if singular_form(word) == 'eye':
                    is_eye_tag = True
                    break
            else:
                continue
        if is_eye_tag:
            eye_tags.append(tag)

    for step_item in tqdm(df_selected_steps.to_dict('records')):
        step = step_item['step']
        for i in range(infer_seed_count):
            step_infer_dir = os.path.join(workdir, 'infer', 'raw', str(step), str(i))
            os.makedirs(step_infer_dir, exist_ok=True)
            step_eval_infer_okay_file = os.path.join(step_infer_dir, '.inferred')
            if os.path.exists(step_eval_infer_okay_file):
                logging.info(f'Step {step}, repeat #{i} already inferred for scale, skipped.')
            else:
                seed = random.randint(0, 1 << 30)
                os.makedirs(step_infer_dir, exist_ok=True)
                logging.info(f'Infer for step {step}, repeat #{i}, seed: {seed} ...')
                pairs, lora_name = infer_with_lora(
                    lora_file=lora_files[step],
                    eye_tags=eye_tags,
                    df_tags=df_tags,
                    seed=seed,
                    batch_size=batch_size,
                    sampler_name=sampler_name,
                    cfg_scale=cfg_scale,
                    steps=steps,
                    firstphase_width=firstphase_width,
                    firstphase_height=firstphase_height,
                    enable_hr=enable_hr,
                    hr_resize_x=hr_resize_x,
                    hr_resize_y=hr_resize_y,
                    denoising_strength=denoising_strength,
                    hr_second_pass_steps=hr_second_pass_steps,
                    hr_upscaler=hr_upscaler,
                    clip_skip=clip_skip,
                    lora_alpha=lora_alpha,
                    enable_adetailer=enable_adetailer,
                    base_model=base_model,
                    extra_tags=[] if not bangumi_style_tag else [bangumi_style_tag],
                )
                for name, image in tqdm(pairs, desc='Save Images'):
                    param_text = image.info.get('parameters').replace(lora_name, trigger_name)
                    sdmeta = parse_sdmeta_from_text(param_text)
                    dst_image_file = os.path.join(step_infer_dir, f'{name}.png')
                    image.save(dst_image_file, pnginfo=sdmeta.pnginfo)

                pathlib.Path(step_eval_infer_okay_file).touch()
