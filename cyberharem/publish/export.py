import json
import logging
import os.path
import shutil
import zipfile
from textwrap import dedent

import pandas as pd

from .convert import convert_to_webui_lora
from .steps import find_steps_in_workdir
from ..dataset.tags import sort_draw_names
from ..infer.draw import _DEFAULT_INFER_MODEL
from ..infer.draw import draw_with_workdir, Drawing


def _make_preview_info(draw: Drawing, n_repeats: int = 2):
    return dedent(f"""
Prompt: {draw.prompt}
Neg Prompt: {draw.neg_prompt}
Width: {draw.width}
Height: {draw.height}
Guidance Scale: {draw.gscale}
Infer Steps: {draw.steps}
N Repeats: {n_repeats}
Seed: {draw.seed}
Safe For Word: {"yes" if draw.sfw else "no"}
    """).lstrip()


def export_workdir(workdir: str, export_dir: str, n_repeats: int = 2,
                   pretrained_model: str = _DEFAULT_INFER_MODEL, clip_skip: int = 1,
                   image_width: int = 512, image_height: int = 768, infer_steps: int = 30):
    name, steps = find_steps_in_workdir(workdir)
    logging.info(f'Starting export trained artifacts of {name!r}, with steps: {steps!r}')

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
                    clip_skip=clip_skip,
                )
            except RuntimeError:
                n_repeats += 1
            else:
                break
        for draw in drawings:
            draw.image.save(os.path.join(preview_dir, f'{draw.name}.png'))
            with open(os.path.join(preview_dir, f'{draw.name}_info.txt'), 'w', encoding='utf-8') as f:
                print(_make_preview_info(draw, n_repeats), file=f)
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

    nsfw_ratio = {name: count * 1.0 / len(steps) for name, count in nsfw_count.items()}
    with open(os.path.join(export_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'name': name,
            'steps': steps,
        }, f, ensure_ascii=False, indent=4)
    with open(os.path.join(export_dir, 'README.md'), 'w', encoding='utf-8') as f:
        print(f'# Lora of {name}', file=f)
        print('', file=f)

        print('This model is trained with [HCP-Diffusion](https://github.com/7eu7d7/HCP-Diffusion). '
              'And the auto-training framework is maintained by '
              '[DeepGHS Team](https://huggingface.co/deepghs).', file=f)
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
