import datetime
import glob
import json
import math
import os
import shutil
import time
import zipfile
from textwrap import dedent
from typing import Optional, Dict

import numpy as np
import pandas as pd
from ditk import logging
from hbutils.random import random_sha1_with_timestamp
from hbutils.scale import size_to_bytes_str
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory
from huggingface_hub import CommitOperationAdd, hf_hub_url
from imgutils.tagging import remove_underline
from tqdm import tqdm

from .export import _GITLFS
from ..eval import eval_for_workdir
from ..infer import find_steps_in_workdir, find_tags_from_workdir
from ..utils import get_hf_client, get_hf_fs, create_safe_toml


def _init_model_repo(repository: str):
    logging.info(f'Initializing repository {repository!r} ...')
    hf_fs = get_hf_fs()
    hf_client = get_hf_client()
    if not hf_fs.exists(f'{repository}/.gitattributes'):
        hf_client.create_repo(repo_id=repository, repo_type='model', exist_ok=True)

    if not hf_fs.exists(f'{repository}/.gitattributes') or \
            '*.png filter=lfs diff=lfs merge=lfs -text' not in hf_fs.read_text(f'{repository}/.gitattributes'):
        logging.info(f'Preparing for lfs attributes of repository {repository!r}.')
        with TemporaryDirectory() as td:
            _git_attr_file = os.path.join(td, '.gitattributes')
            with open(_git_attr_file, 'w', encoding='utf-8') as f:
                print(_GITLFS, file=f)

            operations = [
                CommitOperationAdd(
                    path_in_repo='.gitattributes',
                    path_or_fileobj=_git_attr_file,
                )
            ]

            current_time = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
            commit_message = f'Prepare .gitattributes, on {current_time}'
            logging.info(f'Preparing .gitattributes to repository {repository!r} ...')
            hf_client.create_commit(
                repository,
                operations,
                commit_message=commit_message,
                repo_type='model',
            )


class CCIPTooLowError(Exception):
    pass


def _prepare_for_attempt_dir(workdir: str, info: Dict) -> str:
    new_workdir = os.path.join(
        os.path.dirname(workdir),
        f'{os.path.basename(workdir)}_{random_sha1_with_timestamp()}'
    )
    logging.info(f'Moving workdir {workdir!r} to {new_workdir!r} ...')
    shutil.move(workdir, new_workdir)

    os.makedirs(workdir, exist_ok=True)
    last_attempt_file = os.path.join(workdir, 'last_attempt.json')
    logging.info(f'Writing information to last attempt file {last_attempt_file!r} ...')
    with open(last_attempt_file, 'w') as f:
        json.dump({
            'workdir': os.path.abspath(new_workdir),
            'rel_workdir': os.path.relpath(
                os.path.abspath(new_workdir),
                start=os.path.abspath(workdir),
            ),
            'info': info,
        }, f, indent=4, sort_keys=True, ensure_ascii=False)

    return new_workdir


def deploy_to_huggingface(
        workdir: str, repository: Optional[str] = None, eval_cfgs: Optional[dict] = None,
        steps_batch_size: int = 10, discord_publish: bool = True,
        ccip_check: Optional[float] = 0.72, move_when_check_failed: bool = True,
        force_upload_after_ckpts: Optional[int] = 75,
        ccip_distance_mode: bool = False,
        revision: str = 'main'
):
    with open(os.path.join(workdir, 'meta.json'), 'r') as f:
        meta_info = json.load(f)
    base_model_type = meta_info.get('base_model_type', 'SD1.5')
    train_type = meta_info.get('train_type', 'LoRA')

    logging.info('Starting evaluation before deployment ...')
    eval_for_workdir(workdir, ccip_distance_mode=ccip_distance_mode, **(eval_cfgs or {}))

    df = pd.read_csv(os.path.join(workdir, 'eval', 'metrics.csv'))
    if ccip_check is not None and df['ccip'].max() < ccip_check:
        if force_upload_after_ckpts is None or len(df) < force_upload_after_ckpts:
            if move_when_check_failed:
                _prepare_for_attempt_dir(workdir, info={
                    'reason': 'step_too_low',
                })
            raise CCIPTooLowError(f'CCIP too low, minimum {ccip_check:.3f} required, but {df["ccip"].max():.3f} found.')
        else:
            logging.warning(f'CCIP still too low, minimum {ccip_check:.3f} required, '
                            f'but {df["ccip"].max():.3f} found. '
                            f'Still uploaded due to settings.')

    name = meta_info['name']
    ds_repo = meta_info['dataset']['repository']
    repository = repository or ds_repo
    _init_model_repo(repository)

    df_steps = find_steps_in_workdir(workdir)
    steps = list(df_steps['step'])
    logging.info(f'Steps {steps!r} found.')

    df_tags = find_tags_from_workdir(workdir)
    rtag_names = list(df_tags['name'])
    logging.info(f'RTags {rtag_names!r} found.')

    infos = {item['step']: item for item in
             pd.read_csv(os.path.join(workdir, 'eval', 'metrics.csv')).to_dict('records')}

    def _make_table_for_steps(stps, cur_path='.'):
        columns = ['Step', 'Epoch', 'CCIP' if not ccip_distance_mode else 'C-Diff',
                   'AI Corrupt', 'Bikini Plus', 'Score', 'Download', *rtag_names]

        v_data = []
        for s in stps:
            v_data.append({
                'step': s,
                'epoch': infos[s]["epoch"],
                'ccip': infos[s]["ccip"],
                'aic': infos[s]["aic"],
                'bp': infos[s]["bp"],
                'integrate': infos[s]["integrate"]
            })
        v_df = pd.DataFrame(v_data)
        ccip_max = v_df['ccip'].max()
        aic_max = v_df['aic'].max()
        bp_max = v_df['bp'].max()
        integrate_max = v_df['integrate'].max()

        ret_data = []
        for s in stps:
            download_url = hf_hub_url(
                repo_id=repository, repo_type="model",
                filename=os.path.join(str(s), f"{name}.zip"),
            )
            img_values = []
            for rt in rtag_names:
                png_path = f'{s}/previews/{rt}.png'
                png_path = os.path.relpath(png_path, cur_path)
                img_values.append(f'![{rt}]({png_path})')

            ccip_value = infos[s]["ccip"]
            aic_value = infos[s]["aic"]
            bp_value = infos[s]["bp"]
            integrate_value = infos[s]["integrate"]
            ret_data.append([
                f'{s}',
                infos[s]["epoch"],
                f'**{ccip_value:.3f}**' if np.isclose(ccip_value, ccip_max).item() else f'{ccip_value:.3f}',
                f'**{aic_value:.3f}**' if np.isclose(aic_value, aic_max).item() else f'{aic_value:.3f}',
                f'**{bp_value:.3f}**' if np.isclose(bp_value, bp_max).item() else f'{bp_value:.3f}',
                f'**{integrate_value:.3f}**' if np.isclose(integrate_value,
                                                           integrate_max).item() else f'{integrate_value:.3f}',
                f'[Download]({download_url})',
                *img_values
            ])

        return pd.DataFrame(ret_data, columns=columns)

    with TemporaryDirectory() as td:
        for step_item in tqdm(df_steps.to_dict('records'), desc='Preparing Steps'):
            step = step_item['step']
            logging.info(f'Packing for step {step!r} ...')
            step_dir = os.path.join(td, str(step))
            step_previews_dir = os.path.join(step_dir, 'previews')
            os.makedirs(step_previews_dir, exist_ok=True)
            logging.info(f'Copying images to {step_previews_dir!r} ...')
            for png_file in glob.glob(os.path.join(step_item['workdir'], 'eval', str(step), '*.png')):
                shutil.copyfile(png_file, os.path.join(step_previews_dir, os.path.basename(png_file)))
            shutil.copyfile(
                os.path.join(step_item['workdir'], 'eval', str(step), 'metrics.json'),
                os.path.join(step_dir, 'metrics.json'),
            )
            shutil.copyfile(
                os.path.join(step_item['workdir'], 'eval', str(step), 'details.csv'),
                os.path.join(step_dir, 'details.csv'),
            )

            origin_lora_file = os.path.join(step_item['workdir'], 'kohya', step_item['filename'])
            final_lora_file = os.path.join(step_dir, f'{name}.safetensors')
            logging.info(f'Copy lora file to {final_lora_file!r} ...')
            shutil.copyfile(origin_lora_file, final_lora_file)

            zip_file = os.path.join(step_dir, f'{name}.zip')
            logging.info(f'Packing package {zip_file!r} ...')
            with zipfile.ZipFile(zip_file, 'w') as f:
                f.write(final_lora_file, f'{name}.safetensors')
                for png_file in glob.glob(os.path.join(step_previews_dir, '*.png')):
                    f.write(png_file, os.path.basename(png_file))

        logging.info('Copying full metrics files ...')
        shutil.copyfile(
            os.path.join(workdir, 'eval', 'metrics.csv'),
            os.path.join(td, 'metrics.csv'),
        )
        shutil.copyfile(
            os.path.join(workdir, 'eval', 'metrics_selected.csv'),
            os.path.join(td, 'metrics_selected.csv'),
        )
        shutil.copyfile(
            os.path.join(workdir, 'eval', 'metrics_plot.png'),
            os.path.join(td, 'metrics_plot.png'),
        )

        logging.info('Copying character feature file ...')
        shutil.copyfile(
            os.path.join(workdir, 'features.npy'),
            os.path.join(td, 'features.npy'),
        )

        train_toml_file = os.path.join(td, 'train.toml')
        logging.info('Copying safe train toml file ...')
        create_safe_toml(os.path.join(workdir, 'train.toml'), train_toml_file)

        from ..train.train import TRAIN_MARK
        meta_info['base_model_type'] = base_model_type
        meta_info['train_type'] = train_type
        meta_info['version'] = TRAIN_MARK
        meta_info['time'] = time.time()
        selected_steps = pd.read_csv(os.path.join(td, 'metrics_selected.csv'))['step'].tolist()
        best_step = selected_steps[0]
        meta_info['steps'] = steps
        meta_info['rtags'] = rtag_names
        meta_info['selected_steps'] = selected_steps
        meta_info['best_step'] = best_step

        with open(os.path.join(td, 'meta.json'), 'w') as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=4)
        with open(os.path.join(td, '.gitattributes'), 'w', encoding='utf-8') as f:
            print(_GITLFS, file=f)
        with open(os.path.join(td, 'README.md'), 'w', encoding='utf-8') as f:
            print(f'---', file=f)
            print(f'license: mit', file=f)
            print(f'datasets:', file=f)
            print(f'- {meta_info["dataset"]["repository"]}', file=f)
            if meta_info['bangumi']:
                print(f'- {meta_info["bangumi"]}', file=f)
            print(f'pipeline_tag: text-to-image', file=f)
            print(f'tags:', file=f)
            print(f'- art', file=f)
            print(f'- not-for-all-audiences', file=f)
            print(f'---', file=f)
            print(f'', file=f)

            print(f'# LoRA model of {meta_info["display_name"]}', file=f)
            print(f'', file=f)

            print(f'## What Is This?', file=f)
            print(f'', file=f)
            print(f'This is the LoRA model of waifu {meta_info["display_name"]}.', file=f)
            print(f'', file=f)

            print(f'## How Is It Trained?', file=f)
            print(f'', file=f)
            print(f'* This model is trained with [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts), '
                  f'and the test images are generated with [a1111\'s webui](AUTOMATIC1111/stable-diffusion-webui) and '
                  f'[API sdk](https://github.com/mix1009/sdwebuiapi).', file=f)
            print(f'* The [auto-training framework](https://github.com/deepghs/cyberharem) is maintained by '
                  f'[DeepGHS Team](https://huggingface.co/deepghs).', file=f)

            print(f'The architecture of base model is is `{base_model_type}`.', file=f)
            dataset_info = meta_info['dataset']
            print(f'* Dataset used for training is the `{dataset_info["name"]}` in '
                  f'[{dataset_info["repository"]}](https://huggingface.co/datasets/{dataset_info["repository"]}), '
                  f'which contains {plural_word(dataset_info["size"], "image")}.', file=f)
            if meta_info['bangumi']:
                print(f'* The images in the dataset is auto-cropped from anime videos, '
                      f'more images for other waifus in the same anime can be found in '
                      f'[{meta_info["bangumi"]}](https://huggingface.co/datasets/{meta_info["bangumi"]})', file=f)

            print(f'* **Trigger word is `{name}`.**', file=f)
            if meta_info.get('bangumi_style_name'):
                bangumi_style_name = meta_info['bangumi_style_name']
                print(f'* **Trigger word of anime style is is `{bangumi_style_name}`.**', file=f)
            print(f'* Pruned core tags for this waifu are '
                  f'`{", ".join(map(remove_underline, meta_info["core_tags"]))}`. '
                  f'You can add them to the prompt when some features of waifu '
                  f'(e.g. hair color) are not stable.', file=f)

            train_toml_url = hf_hub_url(repo_id=repository, repo_type='model', filename=f'train.toml')
            print(f'* For more details in training, you can take a look at '
                  f'[training configuration file]({train_toml_url}).', file=f)
            print(f'* For more details in LoRA, you can download it, '
                  f'and read the metadata with a1111\'s webui.', file=f)
            print(f'', file=f)

            print(f'## How to Use It?', file=f)
            print(f'', file=f)
            print(f'After downloading the safetensors files for the specified step, '
                  f'you need to use them like common LoRA.', file=f)
            print(f'', file=f)
            print(f'* Recommended LoRA weight is 0.5-0.85.', file=f)
            print(f'* Recommended trigger word weight is 0.7-1.1.', file=f)
            print(f'', file=f)

            model_url = hf_hub_url(repo_id=repository, repo_type='model', filename=f'{best_step}/{name}.safetensors')
            print(f'For example, if you want to use the model from step {best_step}, '
                  f'you need to download [`{best_step}/{name}.safetensors`]({model_url}) '
                  f'as LoRA. By using this model, you can generate images for the desired characters.', file=f)
            print(f'', file=f)

            print(f'## Which Step Should I Use?', file=f)
            print(f'', file=f)
            print(f'We selected {plural_word(len(selected_steps), "good step")} for you to choose. '
                  f'The best one is step {best_step!r}.', file=f)
            print(f'', file=f)

            all_images = glob.glob(os.path.join(td, '*', 'previews', '*.png'))
            all_images_count = len(all_images)
            all_images_size = sum(map(os.path.getsize, all_images))
            print(f'{plural_word(all_images_count, "image")} '
                  f'({size_to_bytes_str(all_images_size, precision=2)}) '
                  f'were generated for auto-testing.', file=f)
            print(f'', file=f)
            print(f'![Metrics Plot](metrics_plot.png)', file=f)
            print(f'', file=f)

            print(f'Here are the preview of the recommended steps:', file=f)
            print(f'', file=f)
            print(_make_table_for_steps(selected_steps).to_markdown(index=False), file=f)
            print(f'', file=f)

            print(f'## Anything Else?', file=f)
            print(f'', file=f)

            print(dedent(f"""
                    Because the automation of LoRA training always annoys some people. So for the following groups, it is not recommended to use this model and we express regret:
                    1. Individuals who cannot tolerate any deviations from the original character design, even in the slightest detail.
                    2. Individuals who are facing the application scenarios with high demands for accuracy in recreating character outfits.
                    3. Individuals who cannot accept the potential randomness in AI-generated images based on the Stable Diffusion algorithm.
                    4. Individuals who are not comfortable with the fully automated process of training character models using LoRA, or those who believe that training character models must be done purely through manual operations to avoid disrespecting the characters.
                    5. Individuals who finds the generated image content offensive to their values.
                """).strip(), file=f)
            print(f'', file=f)

            print(f'## All Steps', file=f)
            print(f'', file=f)
            print(f'We uploaded the files in all steps. you can check the images, '
                  f'metrics and download them in the following links: ', file=f)
            all_index_dir = os.path.join(td, 'all')
            os.makedirs(all_index_dir, exist_ok=True)
            batch_count = int(math.ceil(len(steps) / steps_batch_size))

            for i in range(batch_count):
                s_steps = steps[::-1][i * steps_batch_size: (i + 1) * steps_batch_size]
                s_table = _make_table_for_steps(s_steps, cur_path=os.path.relpath(all_index_dir, td))
                text = f'Steps From {min(s_steps)} to {max(s_steps)}'
                index_file = os.path.join(all_index_dir, f'{i}.md')
                print(f'* [{text}]({os.path.relpath(index_file, td)})', file=f)

                with open(index_file, 'w') as md_f:
                    print(f'# {text}', file=md_f)
                    print(f'', file=md_f)
                    print(s_table.to_markdown(index=False), file=md_f)
                    print(f'', file=md_f)

        logging.info(f'Uploading files to repository {repository!r} ...')
        upload_directory_as_directory(
            local_directory=td,
            path_in_repo='.',
            repo_id=repository,
            repo_type='model',
            message=f'Upload model for {meta_info["display_name"]}',
            clear=True,
            revision=revision,
        )

    if discord_publish and 'GH_TOKEN' in os.environ:
        from .discord import send_discord_publish_to_github_action
        send_discord_publish_to_github_action(repository)
