import json
import logging
import os.path
import shutil
import time
import zipfile
from textwrap import dedent
from typing import Optional

import pandas as pd

from .convert import convert_to_webui_lora
from .steps import find_steps_in_workdir
from ..dataset.tags import sort_draw_names
from ..infer.draw import _DEFAULT_INFER_MODEL
from ..infer.draw import draw_with_workdir

KNOWN_MODEL_HASHES = {
    'AIARTCHAN/anidosmixV2': 'EB49192009',
    'stablediffusionapi/anything-v5': None,
    'stablediffusionapi/cetusmix': 'B42B09FF12',
    'Meina/MeinaMix_V10': 'D967BCAE4A',
    'Meina/MeinaMix_V11': '54EF3E3610',
}

EXPORT_MARK = 'v1.3'

_GITLFS = dedent("""
*.7z filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.bz2 filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.ftz filter=lfs diff=lfs merge=lfs -text
*.gz filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.lfs.* filter=lfs diff=lfs merge=lfs -text
*.mlmodel filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.rar filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
saved_model/**/* filter=lfs diff=lfs merge=lfs -text
*.tar.* filter=lfs diff=lfs merge=lfs -text
*.tar filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.tgz filter=lfs diff=lfs merge=lfs -text
*.wasm filter=lfs diff=lfs merge=lfs -text
*.xz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.zst filter=lfs diff=lfs merge=lfs -text
*tfevents* filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
""").strip()


def export_workdir(workdir: str, export_dir: str, n_repeats: int = 2,
                   pretrained_model: str = _DEFAULT_INFER_MODEL, clip_skip: int = 2,
                   image_width: int = 512, image_height: int = 768, infer_steps: int = 30,
                   lora_alpha: float = 0.85, sample_method: str = 'DPM++ 2M Karras',
                   model_hash: Optional[str] = None):
    name, steps = find_steps_in_workdir(workdir)
    logging.info(f'Starting export trained artifacts of {name!r}, with steps: {steps!r}')
    model_hash = model_hash or KNOWN_MODEL_HASHES.get(pretrained_model, None)
    if model_hash:
        logging.info(f'Model hash {model_hash!r} detected for model {pretrained_model!r}.')

    d_names = set()
    all_drawings = {}
    nsfw_count = {}
    for step in steps:
        logging.info(f'Exporting for {name}-{step} ...')
        step_dir = os.path.join(export_dir, f'{step}')
        os.makedirs(step_dir, exist_ok=True)

        preview_dir = os.path.join(step_dir, 'previews')
        os.makedirs(preview_dir, exist_ok=True)

        while True:
            try:
                drawings = draw_with_workdir(
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
            img_file = os.path.join(preview_dir, f'{draw.name}.png')
            draw.image.save(img_file, pnginfo=draw.pnginfo)
            all_image_files.append(img_file)

            with open(os.path.join(preview_dir, f'{draw.name}_info.txt'), 'w', encoding='utf-8') as f:
                print(draw.preview_info, file=f)
            d_names.add(draw.name)
            all_drawings[(draw.name, step)] = draw
            if not draw.sfw:
                nsfw_count[draw.name] = nsfw_count.get(draw.name, 0) + 1

        pt_file = os.path.join(workdir, 'ckpts', f'{name}-{step}.pt')
        unet_file = os.path.join(workdir, 'ckpts', f'unet-{step}.safetensors')
        text_encoder_file = os.path.join(workdir, 'ckpts', f'text_encoder-{step}.safetensors')
        raw_dir = os.path.join(step_dir, 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        shutil.copyfile(pt_file, os.path.join(raw_dir, os.path.basename(pt_file)))
        shutil.copyfile(unet_file, os.path.join(raw_dir, os.path.basename(unet_file)))
        shutil.copyfile(text_encoder_file, os.path.join(raw_dir, os.path.basename(text_encoder_file)))

        shutil.copyfile(pt_file, os.path.join(step_dir, f'{name}.pt'))
        convert_to_webui_lora(unet_file, text_encoder_file, os.path.join(step_dir, f'{name}.safetensors'))
        with zipfile.ZipFile(os.path.join(step_dir, f'{name}.zip'), 'w') as zf:
            zf.write(os.path.join(step_dir, f'{name}.pt'), f'{name}.pt')
            zf.write(os.path.join(step_dir, f'{name}.safetensors'), f'{name}.safetensors')
            for img_file in all_image_files:
                zf.write(img_file, os.path.basename(img_file))

    nsfw_ratio = {name: count * 1.0 / len(steps) for name, count in nsfw_count.items()}
    with open(os.path.join(export_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'name': name,
            'steps': steps,
            'mark': EXPORT_MARK,
            'time': time.time(),
        }, f, ensure_ascii=False, indent=4)
    with open(os.path.join(export_dir, '.gitattributes'), 'w', encoding='utf-8') as f:
        print(_GITLFS, file=f)
    with open(os.path.join(export_dir, 'README.md'), 'w', encoding='utf-8') as f:
        print(f'# Lora of {name}', file=f)
        print('', file=f)

        print('This model is trained with [HCP-Diffusion](https://github.com/7eu7d7/HCP-Diffusion). '
              'And the auto-training framework is maintained by '
              '[DeepGHS Team](https://huggingface.co/deepghs).', file=f)
        print('', file=f)

        print('The base model used during training is [NAI](https://huggingface.co/deepghs/animefull-latest), '
              f'and the base model used for generating preview images is '
              f'[{pretrained_model}](https://huggingface.co/{pretrained_model}).', file=f)
        print('', file=f)

        print(f'After downloading the pt and safetensors files for the specified step, '
              f'you need to use them simultaneously. The pt file will be used as an embedding, '
              f'while the safetensors file will be loaded for Lora.', file=f)
        print('', file=f)
        print(f'For example, if you want to use the model from step {steps[-1]}, '
              f'you need to download `{steps[-1]}/{name}.pt` as the embedding and '
              f'`{steps[-1]}/{name}.safetensors` for loading Lora. '
              f'By using both files together, you can generate images for the desired characters.', file=f)
        print('', file=f)

        print(f'**The trigger word is `{name}`.**', file=f)
        print('', file=f)

        print(dedent("""
For the following groups, it is not recommended to use this model and we express regret:
1. Individuals who cannot tolerate any deviations from the original character design, even in the slightest detail.
2. Individuals who are facing the application scenarios with high demands for accuracy in recreating character outfits.
3. Individuals who cannot accept the potential randomness in AI-generated images based on the Stable Diffusion algorithm.
4. Individuals who are not comfortable with the fully automated process of training character models using LoRA, or those who believe that training character models must be done purely through manual operations to avoid disrespecting the characters.
5. Individuals who finds the generated image content offensive to their values.
        """).strip(), file=f)
        print('', file=f)

        print(f'These are available steps:', file=f)
        print('', file=f)

        d_names = sort_draw_names(list(d_names))
        columns = ['Steps', 'Download', *d_names]
        t_data = []

        for step in steps[::-1]:
            d_mds = []
            for dname in d_names:
                file = os.path.join(str(step), 'previews', f'{dname}.png')
                if (dname, step) in all_drawings:
                    if nsfw_ratio.get(dname, 0.0) < 0.35:
                        d_mds.append(f'![{dname}-{step}]({file})')
                    else:
                        d_mds.append(f'[<NSFW, click to see>]({file})')
                else:
                    d_mds.append('')

            t_data.append((str(step), f'[Download]({step}/{name}.zip)', *d_mds))

        df = pd.DataFrame(columns=columns, data=t_data)
        print(df.to_markdown(index=False), file=f)
        print('', file=f)
