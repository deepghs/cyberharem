import json
import logging
import os
from typing import Optional

from hbutils.system import TemporaryDirectory
from huggingface_hub import hf_hub_url
from tqdm.auto import tqdm

from .draw import _DEFAULT_INFER_MODEL, draw_images_for_workdir
from ..dataset import save_recommended_tags
from ..utils import get_hf_fs, download_file


def draw_to_directory(workdir: str, export_dir: str, step: int, n_repeats: int = 2,
                      pretrained_model: str = _DEFAULT_INFER_MODEL, clip_skip: int = 2,
                      image_width: int = 512, image_height: int = 768, infer_steps: int = 30,
                      lora_alpha: float = 0.85, sample_method: str = 'DPM++ 2M Karras',
                      model_hash: Optional[str] = None):
    from ..publish.export import KNOWN_MODEL_HASHES
    model_hash = model_hash or KNOWN_MODEL_HASHES.get(pretrained_model)
    os.makedirs(export_dir, exist_ok=True)

    while True:
        try:
            drawings = draw_images_for_workdir(
                workdir, model_steps=step, n_repeats=n_repeats,
                pretrained_model=pretrained_model,
                width=image_width, height=image_height, infer_steps=infer_steps,
                lora_alpha=lora_alpha, clip_skip=clip_skip, sample_method=sample_method,
                model_hash=model_hash,
            )
        except RuntimeError:
            n_repeats += 1
        else:
            break

    all_image_files = []
    for draw in drawings:
        img_file = os.path.join(export_dir, f'{draw.name}.png')
        draw.image.save(img_file, pnginfo=draw.pnginfo)
        all_image_files.append(img_file)

        with open(os.path.join(export_dir, f'{draw.name}_info.txt'), 'w', encoding='utf-8') as f:
            print(draw.preview_info, file=f)


def draw_with_repo(repository: str, export_dir: str, step: Optional[int] = None, n_repeats: int = 2,
                   pretrained_model: str = _DEFAULT_INFER_MODEL, clip_skip: int = 2,
                   image_width: int = 512, image_height: int = 768, infer_steps: int = 30,
                   lora_alpha: float = 0.85, sample_method: str = 'DPM++ 2M Karras',
                   model_hash: Optional[str] = None):
    from ..publish import find_steps_in_workdir

    hf_fs = get_hf_fs()
    if not hf_fs.exists(f'{repository}/meta.json'):
        raise ValueError(f'Invalid repository or no model found - {repository!r}.')

    logging.info(f'Model repository {repository!r} found.')
    meta = json.loads(hf_fs.read_text(f'{repository}/meta.json'))
    step = step or meta['best_step']
    logging.info(f'Using step {step} ...')

    with TemporaryDirectory() as workdir:
        logging.info('Downloading models ...')
        for f in tqdm(hf_fs.glob(f'{repository}/{step}/raw/*')):
            rel_file = os.path.relpath(f, repository)
            local_file = os.path.join(workdir, 'ckpts', os.path.basename(rel_file))
            if os.path.dirname(local_file):
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
            download_file(
                hf_hub_url(repository, filename=rel_file),
                local_file
            )

        logging.info(f'Regenerating tags for {workdir!r} ...')
        pt_name, _ = find_steps_in_workdir(workdir)
        game_name = pt_name.split('_')[-1]
        name = '_'.join(pt_name.split('_')[:-1])

        from gchar.games.dispatch.access import GAME_CHARS
        if game_name in GAME_CHARS:
            ch_cls = GAME_CHARS[game_name]
            ch = ch_cls.get(name)
        else:
            ch = None

        if ch is None:
            source = repository
        else:
            source = ch

        logging.info(f'Regenerate tags for {source!r}, on {workdir!r}.')
        save_recommended_tags(source, name=pt_name, workdir=workdir, ds_size=meta["dataset"]['type'])

        logging.info('Drawing ...')
        draw_to_directory(
            workdir, export_dir, step,
            n_repeats, pretrained_model, clip_skip, image_width, image_height, infer_steps,
            lora_alpha, sample_method, model_hash
        )
