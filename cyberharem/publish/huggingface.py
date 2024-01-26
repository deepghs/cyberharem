import datetime
import glob
import json
import math
import os
import shutil
import time
import zipfile
from textwrap import dedent
from typing import Optional

import numpy as np
import pandas as pd
from ditk import logging
from hbutils.scale import size_to_bytes_str
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory
from huggingface_hub import CommitOperationAdd, hf_hub_url
from imgutils.sd import get_sdmeta_from_image
from tqdm import tqdm

from .convert import convert_to_webui_lora, pack_to_bundle_lora
from .export import _GITLFS, EXPORT_MARK
from ..eval import eval_for_workdir, list_steps
from ..infer.draw import list_rtag_names
from ..utils import get_hf_client, get_hf_fs


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


_BLACKLIST_WORDS = ['dir', 'cache']


def _dict_prune(d):
    if isinstance(d, dict):
        return {
            key: _dict_prune(value)
            for key, value in d.items()
            if not any(bw in key for bw in _BLACKLIST_WORDS)
        }
    elif isinstance(d, list):
        return [_dict_prune(item) for item in d]
    else:
        return d


def deploy_to_huggingface(workdir: str, repository: Optional[str] = None, eval_cfgs: Optional[dict] = None,
                          steps_batch_size: int = 10):
    logging.info('Starting evaluation before deployment ...')
    eval_for_workdir(workdir, **(eval_cfgs or {}))

    with open(os.path.join(workdir, 'meta.json'), 'r') as f:
        meta_info = json.load(f)

    name = meta_info['name']
    ds_repo = meta_info['dataset']['repository']
    repository = repository or ds_repo
    _init_model_repo(repository)

    steps = list_steps(workdir)
    logging.info(f'Steps {steps!r} found.')
    rtag_names = list_rtag_names(workdir)
    logging.info(f'RTags {rtag_names!r} found.')

    infos = {item['step']: item for item in
             pd.read_csv(os.path.join(workdir, 'eval', 'metrics.csv')).to_dict('records')}

    def _make_table_for_steps(stps, cur_path='.'):
        columns = ['Step', 'Epoch', 'CCIP', 'AI Corrupt', 'Bikini Plus', 'Score', 'Download', *rtag_names]

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
                f'{s}[{os.path.relpath(f"{s}", cur_path)}]',
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
        for step in tqdm(steps, desc='Preparing Steps'):
            logging.info(f'Packing for step {step!r} ...')
            step_dir = os.path.join(td, str(step))
            step_previews_dir = os.path.join(step_dir, 'previews')
            os.makedirs(step_previews_dir, exist_ok=True)
            logging.info(f'Copying images to {step_previews_dir!r} ...')
            for png_file in glob.glob(os.path.join(workdir, 'eval', str(step), '*.png')):
                shutil.copyfile(png_file, os.path.join(step_previews_dir, os.path.basename(png_file)))
            shutil.copyfile(
                os.path.join(workdir, 'eval', str(step), 'metrics.json'),
                os.path.join(step_dir, 'metrics.json'),
            )
            shutil.copyfile(
                os.path.join(workdir, 'eval', str(step), 'details.csv'),
                os.path.join(step_dir, 'details.csv'),
            )

            step_raw_dir = os.path.join(step_dir, 'raw')
            os.makedirs(step_raw_dir, exist_ok=True)
            logging.info(f'Copying raw model files to {step_raw_dir!r} ...')
            for model_file in glob.glob(os.path.join(workdir, 'ckpts', f'*-{step}.*')):
                shutil.copyfile(model_file, os.path.join(step_raw_dir, os.path.basename(model_file)))

            unet_file = os.path.join(step_raw_dir, f'unet-{step}.safetensors')
            pt_file = os.path.join(step_raw_dir, f'{name}-{step}.pt')
            raw_lora_file = os.path.join(step_dir, f'{name}_raw.safetensors')
            logging.info(f'Dumping raw LoRA to {raw_lora_file!r}...')
            convert_to_webui_lora(
                lora_path=unet_file,
                lora_path_te=None,
                dump_path=raw_lora_file,
            )

            bundled_lora_file = os.path.join(step_dir, f'{name}.safetensors')
            logging.info(f'Creating bundled LoRA to {bundled_lora_file!r} ...')
            pack_to_bundle_lora(
                lora_model=raw_lora_file,
                embeddings={name: pt_file},
                bundle_lora_path=bundled_lora_file,
            )
            shutil.copyfile(pt_file, os.path.join(step_dir, f'{name}.pt'))

            zip_file = os.path.join(step_dir, f'{name}.zip')
            logging.info(f'Packing package {zip_file!r} ...')
            with zipfile.ZipFile(zip_file, 'w') as f:
                f.write(bundled_lora_file, f'{name}.safetensors')
                f.write(pt_file, f'{name}.pt')
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

        meta_info['version'] = EXPORT_MARK
        meta_info['time'] = time.time()
        selected_steps = pd.read_csv(os.path.join(td, 'metrics_selected.csv'))['step'].tolist()
        best_step = selected_steps[0]
        meta_info['steps'] = steps
        meta_info['rtags'] = rtag_names
        meta_info['selected_steps'] = selected_steps
        meta_info['best_step'] = best_step

        with open(os.path.join(td, 'meta.json'), 'w') as f:
            json.dump(_dict_prune(meta_info), f, ensure_ascii=False, indent=4)

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

                print(f'# Lora of {meta_info["display_name"]}', file=f)
                print(f'', file=f)

                print(f'## What Is This?', file=f)
                print(f'', file=f)
                print(f'This is the LoRA model of waifu {meta_info["display_name"]}.', file=f)
                print(f'', file=f)

                print(f'## How Is It Trained?', file=f)
                print(f'', file=f)
                print(f'* This model is trained with [HCP-Diffusion](https://github.com/7eu7d7/HCP-Diffusion).', file=f)
                print(f'* The [auto-training framework](https://github.com/deepghs/cyberharem) is maintained by '
                      f'[DeepGHS Team](https://huggingface.co/deepghs).', file=f)

                train_pretrained_model = meta_info['train']['model']["pretrained_model_name_or_path"]
                print(f'* The base model used for training is '
                      f'[{train_pretrained_model}](https://huggingface.co/{train_pretrained_model}).', file=f)

                dataset_info = meta_info['dataset']
                print(f'* Dataset used for training is the `{dataset_info["name"]}` in '
                      f'[{dataset_info["repository"]}](https://huggingface.co/datasets/{dataset_info["repository"]}), '
                      f'which contains {plural_word(dataset_info["size"], "image")}.', file=f)
                if meta_info['bangumi']:
                    print(f'* The images in the dataset is cropped from anime videos, '
                          f'more images for other waifus in the same anime can be found in '
                          f'[{meta_info["bangumi"]}](https://huggingface.co/datasets/{meta_info["bangumi"]})', file=f)

                ds_res = meta_info["train"]["dataset"]["resolution"]
                print(f'* Batch size is {meta_info["train"]["dataset"]["bs"]}, '
                      f'resolution is {ds_res}x{ds_res}, '
                      f'clustering into {plural_word(meta_info["train"]["dataset"]["num_bucket"], "bucket")}.', file=f)

                reg_res = meta_info["train"]["reg_dataset"]["resolution"]
                print(f'* Batch size for regularization dataset is {meta_info["train"]["reg_dataset"]["bs"]}, '
                      f'resolution is {reg_res}x{reg_res}, '
                      f'clustering into {plural_word(meta_info["train"]["reg_dataset"]["num_bucket"], "bucket")}.',
                      file=f)
                print(f'* Trained for {plural_word(meta_info["train"]["train"]["train_steps"], "step")}, '
                      f'{plural_word(len(steps), "checkpoint")} were saved.', file=f)
                print(f'* **Trigger word is `{name}`.**', file=f)
                print(f'* Pruned core tags for this waifu are `{", ".join(meta_info["core_tags"])}`. '
                      f'You do NOT have to add them in your prompts.', file=f)
                print(f'', file=f)

                print(f'## How to Use It?', file=f)
                print(f'', file=f)
                print(f'### If You Are Using A1111 WebUI v1.7+', file=f)
                print(f'', file=f)
                print(f'**Just use it like the classic LoRA**. '
                      f'The LoRA we provided are bundled with the embedding file.', file=f)
                print(f'', file=f)
                print(f'### If You Are Using A1111 WebUI v1.6 or Lower', file=f)
                print(f'', file=f)
                print(f'After downloading the pt and safetensors files for the specified step, '
                      f'you need to use them simultaneously. The pt file will be used as an embedding, '
                      f'while the safetensors file will be loaded for Lora.', file=f)
                print(f'', file=f)

                pt_url = hf_hub_url(repo_id=repository, repo_type='model', filename=f'{best_step}/{name}.pt')
                lora_url = hf_hub_url(repo_id=repository, repo_type='model', filename=f'{best_step}/{name}.safetensors')
                print(f'For example, if you want to use the model from step {best_step}, '
                      f'you need to download [`{best_step}/{name}.pt`]({pt_url}) as the embedding and '
                      f'[`{best_step}/{name}.safetensors`]({lora_url}) for loading Lora. '
                      f'By using both files together, you can generate images for the desired characters.', file=f)
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

                sample_meta = get_sdmeta_from_image(glob.glob(os.path.join(workdir, 'eval', '*', '*.png'))[0])
                infer_pretrained_model = sample_meta.parameters['Model']
                print(f'The base model used for generating preview images is '
                      f'[{infer_pretrained_model}](https://huggingface.co/{infer_pretrained_model}).', file=f)
                print(f'', file=f)

                print(f'Here are the preview of the recommended steps:', file=f)
                print(f'', file=f)
                print(_make_table_for_steps(selected_steps).to_markdown(index=False), file=f)
                print(f'', file=f)

                print(f'## Anything Else?', file=f)
                print(f'', file=f)

                print(dedent("""
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
                    s_table = _make_table_for_steps(
                        steps[::-1][i * steps_batch_size: (i + 1) * steps_batch_size],
                        cur_path=os.path.relpath(all_index_dir, td)
                    )
                    text = f'Steps From {s_table["Step"].min()} to {s_table["Step"].max()}'
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
        )
