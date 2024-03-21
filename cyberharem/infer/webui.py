import glob
import json
import logging
import os.path
import re
from typing import Optional, List

import pandas as pd
from hbutils.random import random_sha1_with_timestamp
from hbutils.string import singular_form
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
    global _WEBUI_LORA_MOCK
    logging.info(f'Setting webui local directory {webui_local_dir!r} ...')
    _WEBUI_LORA_MOCK = LocalLoraMock(webui_local_dir)


def _get_webui_lora_mock() -> LoraMock:
    return _WEBUI_LORA_MOCK


def infer_with_lora(
        lora_file: str, eyes_tags: List[str], df_tags: pd.DataFrame,
        batch_size=16, sampler_name='DPM++ 2M Karras', cfg_scale=7, steps=30,
        firstphase_width=512, firstphase_height=768, hr_resize_x=832, hr_resize_y=1216,
        denoising_strength=0.6, hr_second_pass_steps=20, hr_upscaler='R-ESRGAN 4x+ Anime6B',
):
    mock = _get_webui_lora_mock()
    client = _get_webui_client()
    lora_name = mock.mock_lora(lora_file)
    try:
        suffix = f'<lora:{lora_name}:1>'

        result = client.txt2img(
            prompt=p,
            negative_prompt='(worst quality, low quality:1.40), (zombie, sketch, interlocked fingers, comic:1.10), (full body:1.10), lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, white border, (english text, chinese text:1.05), (censored, mosaic censoring, bar censor:1.20)',
            alwayson_scripts={
                'Dynamic Prompts v2.17.1': {
                    "args": [
                        True,
                        True,
                    ]
                }
            },
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
            seed=636480265,
            enable_hr=True,
            override_settings={
            },
            adetailer=[
                ADetailer(
                    ad_model='face_yolov8n.pt',
                    ad_prompt='best eyes, masterpiece, best quality, extremely detailed, 8killustration, '
                              'beautiful illustration, beautiful eyes, extremely detailed eyes, shiny eyes, '
                              'lively eyes, livid eyes',
                    ad_denoising_strength=denoising_strength,
                ),
                ADetailer(ad_model='None'),
            ],

        )

    finally:
        mock.unmock_lora(lora_name)


def infer_with_workdir(workdir: str):
    df_steps = find_steps_in_workdir(workdir)
    logging.info(f'Available steps: {len(df_steps)}\n'
                 f'{df_steps}')

    df_tags = find_tags_from_workdir(workdir)
    logging.info(f'Available prompts: {len(df_tags)}\n'
                 f'{df_tags}')

    with open(os.path.join(workdir, 'meta.json')) as f:
        meta = json.load(f)
    name = meta['name']
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
