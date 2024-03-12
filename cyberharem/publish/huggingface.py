import datetime
import glob
import json
import math
import os
import shutil
import textwrap
import time
import zipfile
from textwrap import dedent
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from discord_webhook import DiscordWebhook
from ditk import logging
from hbutils.scale import size_to_bytes_str
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory, download_archive_as_directory, download_file_to_file
from huggingface_hub import CommitOperationAdd, hf_hub_url
from imgutils.sd import get_sdmeta_from_image
from imgutils.validate import anime_rating
from tqdm import tqdm

from .convert import convert_to_webui_lora, pack_to_bundle_lora, convert_to_webui_lycoris
from .export import _GITLFS, EXPORT_MARK
from ..eval import eval_for_workdir, list_steps
from ..infer.draw import list_rtag_names, _DEFAULT_INFER_CFG_FILE_LORA, _DEFAULT_INFER_CFG_FILE_LOKR, \
    _DEFAULT_INFER_CFG_FILE_LORA_SIMPLE, _DEFAULT_INFER_CFG_FILE_LOKR_PIVOTAL
from ..utils import get_hf_client, get_hf_fs
from ..utils.ghaction import GithubActionClient


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


def check_with_train_type(train_type: str = 'Pivotal LoRA') -> Tuple[bool, bool]:
    if train_type == 'Pivotal LoRA':
        is_pivotal = True
        is_lycoris = False
    elif train_type == 'LoRA':
        is_pivotal = False
        is_lycoris = False
    elif train_type == 'LoKr':
        is_pivotal = False
        is_lycoris = True
    elif train_type == 'Pivotal Lokr':
        is_pivotal = True
        is_lycoris = True
    else:
        raise f'Train type not supported - {train_type!r}.'

    return is_pivotal, is_lycoris


def deploy_to_huggingface(workdir: str, repository: Optional[str] = None, eval_cfgs: Optional[dict] = None,
                          steps_batch_size: int = 10, discord_publish: bool = True, enable_bundle: bool = True):
    with open(os.path.join(workdir, 'meta.json'), 'r') as f:
        meta_info = json.load(f)
    base_model_type = meta_info.get('base_model_type', 'SD1.5')
    train_type = meta_info.get('train_type', 'Pivotal LoRA')

    logging.info('Starting evaluation before deployment ...')
    if base_model_type == 'SD1.5':
        is_pivotal, is_lycoris = check_with_train_type(train_type)
        if is_pivotal:
            preset_eval_cfgs = dict(
                cfg_file=_DEFAULT_INFER_CFG_FILE_LORA if not is_lycoris else _DEFAULT_INFER_CFG_FILE_LOKR_PIVOTAL,
                model_tag='lora' if not is_lycoris else 'lokr',
            )
        else:
            preset_eval_cfgs = dict(
                cfg_file=_DEFAULT_INFER_CFG_FILE_LORA_SIMPLE if not is_lycoris else _DEFAULT_INFER_CFG_FILE_LOKR,
                model_tag='lora' if not is_lycoris else 'lokr',
            )
    else:
        raise f'Models other than SD1.5 not supported yet - {base_model_type!r}.'
    if not is_pivotal:
        logging.warning('No pivotal tuning required, so bundle is disabled.')
        enable_bundle = False

    eval_for_workdir(workdir, **{**preset_eval_cfgs, **(eval_cfgs or {})})

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

            if is_lycoris:
                unet_file = os.path.join(step_raw_dir, f'unet-lokr-{step}.safetensors')
                if is_pivotal:
                    pt_file = os.path.join(step_raw_dir, f'{name}-{step}.pt')
                    text_encoder_file = None
                else:
                    pt_file = None
                    text_encoder_file = os.path.join(step_raw_dir, f'text_encoder-lokr-{step}.safetensors')
            else:
                unet_file = os.path.join(step_raw_dir, f'unet-{step}.safetensors')
                if is_pivotal:
                    pt_file = os.path.join(step_raw_dir, f'{name}-{step}.pt')
                    text_encoder_file = None
                else:
                    pt_file = None
                    text_encoder_file = os.path.join(step_raw_dir, f'text_encoder-{step}.safetensors')

            raw_lora_file = os.path.join(step_dir, f'{name}_raw.safetensors')
            if not is_lycoris:
                logging.info(f'Dumping raw LoRA to {raw_lora_file!r}...')
                convert_to_webui_lora(
                    lora_path=unet_file,
                    lora_path_te=text_encoder_file,
                    dump_path=raw_lora_file,
                )
            else:
                logging.info(f'Dumping raw LyCORIS to {raw_lora_file!r}...')
                convert_to_webui_lycoris(
                    lycoris_path=unet_file,
                    lycoris_path_te=text_encoder_file,
                    dump_path=raw_lora_file,
                )

            final_lora_file = os.path.join(step_dir, f'{name}.safetensors')
            if enable_bundle:
                logging.info(f'Creating bundled LoRA to {final_lora_file!r} ...')
                pack_to_bundle_lora(
                    lora_model=raw_lora_file,
                    embeddings={name: pt_file},
                    bundle_lora_path=final_lora_file,
                )
            else:
                logging.info(f'No bundle required, just move lora file to {final_lora_file!r} ...')
                shutil.move(raw_lora_file, final_lora_file)
            if is_pivotal:
                shutil.copyfile(pt_file, os.path.join(step_dir, f'{name}.pt'))

            zip_file = os.path.join(step_dir, f'{name}.zip')
            logging.info(f'Packing package {zip_file!r} ...')
            with zipfile.ZipFile(zip_file, 'w') as f:
                f.write(final_lora_file, f'{name}.safetensors')
                if is_pivotal:
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

        meta_info['base_model_type'] = base_model_type
        meta_info['train_type'] = train_type
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

                print(f'# {"LyCORIS" if is_lycoris else "LoRA"} model of {meta_info["display_name"]}', file=f)
                print(f'', file=f)

                print(f'## What Is This?', file=f)
                print(f'', file=f)
                print(f'This is the {"LyCORIS" if is_lycoris else "LoRA"} model of '
                      f'waifu {meta_info["display_name"]}.', file=f)
                print(f'', file=f)

                print(f'## How Is It Trained?', file=f)
                print(f'', file=f)
                print(f'* This model is trained with [HCP-Diffusion](https://github.com/7eu7d7/HCP-Diffusion).', file=f)
                print(f'* The [auto-training framework](https://github.com/deepghs/cyberharem) is maintained by '
                      f'[DeepGHS Team](https://huggingface.co/deepghs).', file=f)

                train_pretrained_model = meta_info['train']['model']["pretrained_model_name_or_path"]
                print(f'* The base model used for training is '
                      f'[{train_pretrained_model}](https://huggingface.co/{train_pretrained_model}). '
                      f'The architecture is `{base_model_type}`.', file=f)
                if is_pivotal:
                    print(f'* This model is pivotal-tuned, so it should have 2 files -- '
                          f'a {"LyCORIS" if is_lycoris else "LoRA"} file and a embedding file.')

                dataset_info = meta_info['dataset']
                print(f'* Dataset used for training is the `{dataset_info["name"]}` in '
                      f'[{dataset_info["repository"]}](https://huggingface.co/datasets/{dataset_info["repository"]}), '
                      f'which contains {plural_word(dataset_info["size"], "image")}.', file=f)
                if meta_info['bangumi']:
                    print(f'* The images in the dataset is auto-cropped from anime videos, '
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
                      f'{plural_word(len(steps), "checkpoint")} were saved and evaluated.', file=f)
                print(f'* **Trigger word is `{name}`.**', file=f)
                print(f'* Pruned core tags for this waifu are `{", ".join(meta_info["core_tags"])}`. '
                      f'You can add them to the prompt when some features of waifu '
                      f'(e.g. hair color) are not stable.', file=f)
                print(f'', file=f)

                print(f'## How to Use It?', file=f)
                print(f'', file=f)
                if is_pivotal:
                    if enable_bundle:
                        print(f'### If You Are Using A1111 WebUI v1.7+', file=f)
                        print(f'', file=f)
                        print(f'**Just use it like the classic {"LyCORIS" if is_lycoris else "LoRA"}**. '
                              f'The {"LyCORIS" if is_lycoris else "LoRA"} '
                              f'we provided are bundled with the embedding file.', file=f)
                        print(f'', file=f)
                        print(f'### If You Are Using A1111 WebUI v1.6 or Lower', file=f)
                        print(f'', file=f)
                    print(f'After downloading the pt and safetensors files for the specified step, '
                          f'you need to use them simultaneously. The pt file will be used as an embedding, '
                          f'while the safetensors file will be loaded for {"LyCORIS" if is_lycoris else "LoRA"}.',
                          file=f)
                    print(f'', file=f)
                else:
                    print(f'After downloading the safetensors files for the specified step, '
                          f'you need to use them like common {"LyCORIS" if is_lycoris else "LoRA"}.',
                          file=f)
                    print(f'', file=f)

                pt_url = hf_hub_url(repo_id=repository, repo_type='model', filename=f'{best_step}/{name}.pt')
                model_url = hf_hub_url(repo_id=repository, repo_type='model',
                                       filename=f'{best_step}/{name}.safetensors')
                if enable_bundle:
                    print(f'For example, if you want to use the model from step {best_step}, '
                          f'you need to download [`{best_step}/{name}.pt`]({pt_url}) as the embedding and '
                          f'[`{best_step}/{name}.safetensors`]({model_url}) '
                          f'for loading {"LyCORIS" if is_lycoris else "LoRA"}. '
                          f'By using both files together, you can generate images for the desired characters.', file=f)
                else:
                    print(f'For example, if you want to use the model from step {best_step}, '
                          f'you need to download [`{best_step}/{name}.safetensors`]({model_url}) '
                          f'as {"LyCORIS" if is_lycoris else "LoRA"}. '
                          f'By using this {"LyCORIS" if is_lycoris else "LoRA"} model, '
                          f'you can generate images for the desired characters.', file=f)
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

                print(dedent(f"""
                    Because the automation of {"LyCORIS" if is_lycoris else "LoRA"} training always annoys some people. So for the following groups, it is not recommended to use this model and we express regret:
                    1. Individuals who cannot tolerate any deviations from the original character design, even in the slightest detail.
                    2. Individuals who are facing the application scenarios with high demands for accuracy in recreating character outfits.
                    3. Individuals who cannot accept the potential randomness in AI-generated images based on the Stable Diffusion algorithm.
                    4. Individuals who are not comfortable with the fully automated process of training character models using {"LyCORIS" if is_lycoris else "LoRA"}, or those who believe that training character models must be done purely through manual operations to avoid disrespecting the characters.
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
        )

    if discord_publish and 'GH_TOKEN' in os.environ:
        send_discord_publish_to_github_action(repository)


def send_discord_publish_to_github_action(repository: str):
    client = GithubActionClient()
    client.create_workflow_run(
        'deepghs/cyberharem',
        'Discord HF Publish',
        data={
            'repository': repository,
        }
    )


def publish_to_discord(repository: str, max_cnt: Optional[int] = None):
    hf_fs = get_hf_fs()
    meta_info = json.loads(hf_fs.read_text(f'{repository}/meta.json'))
    step = meta_info['best_step']
    name = meta_info['name']
    dataset_size = meta_info['dataset']['size']
    bs = meta_info['train']['dataset']['bs']

    base_model_type = meta_info.get('base_model_type', 'SD1.5')
    train_type = meta_info.get('train_type', 'Pivotal LoRA')

    if base_model_type == 'SD1.5':
        is_pivotal, is_lycoris = check_with_train_type(train_type)
    else:
        raise f'Models other than SD1.5 not supported yet - {base_model_type!r}.'

    with TemporaryDirectory() as td:
        download_archive_as_directory(
            local_directory=td,
            file_in_repo=f'{step}/{name}.zip',
            repo_id=repository,
            repo_type='model',
            hf_token=os.environ.get('HF_TOKEN'),
        )

        local_metrics_plot_png = os.path.join(td, f'metrics_plot.png')
        download_file_to_file(
            local_file=local_metrics_plot_png,
            file_in_repo=f'metrics_plot.png',
            repo_id=repository,
            repo_type='model',
            hf_token=os.environ.get('HF_TOKEN'),
        )

        model_files = []
        lora_file = f'{name}.safetensors'
        lora_path = os.path.join(td, lora_file)
        if os.path.exists(lora_path):
            model_files.append(lora_path)
        pt_file = f'{name}.pt'
        pt_path = os.path.join(td, pt_file)
        if os.path.exists(pt_path):
            model_files.append(pt_path)

        details_csv_file = os.path.join(td, 'details.csv')
        download_file_to_file(
            local_file=details_csv_file,
            file_in_repo=f'{step}/details.csv',
            repo_id=repository,
            repo_type='model',
            hf_token=os.environ.get('HF_TOKEN'),
        )

        from .civitai import _detect_face_value

        metrics_info = json.loads(hf_fs.read_text(f'{repository}/{step}/metrics.json'))
        df = pd.read_csv(details_csv_file)
        df['level'] = [
            2 if 'smile' in img_file or 'portrait' in img_file else (
                1 if 'pattern' in img_file else 0
            )
            for img_file in tqdm(df['image'], desc='Calculating face area')
        ]
        df['face'] = [
            _detect_face_value(os.path.join(td, img_file))
            for img_file in tqdm(df['image'], desc='Calculating face area')
        ]
        df['rating'] = [
            anime_rating(os.path.join(td, img_file))[0]
            for img_file in tqdm(df['image'], desc='Detecting rating')
        ]

        df = df[df['ccip'] >= (metrics_info['ccip'] - 0.05)]
        df = df[df['bp'] >= (metrics_info['bp'] - 0.05)]
        df = df[df['aic'] >= max(metrics_info['aic'] - 0.3, metrics_info['aic'] * 0.5)]

        df['ccip_x'] = np.round(df['ccip'] * 30) / 30.0
        df['face_x'] = np.round(df['face'] * 20) / 20.0
        df = df[df['rating'] != 'r18']
        df = df.sort_values(by=['ccip_x', 'level', 'face_x'], ascending=False)
        if max_cnt is not None:
            df = df[:max_cnt]

        hf_url = f'https://huggingface.co/{repository}'
        dataset_info = meta_info['dataset']
        train_pretrained_model = meta_info['train']['model']["pretrained_model_name_or_path"]
        ds_res = meta_info["train"]["dataset"]["resolution"]
        reg_res = meta_info["train"]["reg_dataset"]["resolution"]
        webhook = DiscordWebhook(
            url=os.environ['DC_MODEL_WEBHOOK'],
            content=textwrap.dedent(f"""
                {"LyCORIS" if is_lycoris else "LoRA"} Model of `{meta_info['display_name']}` has been published to huggingface repository: {hf_url}.
                * **Trigger word is `{name}`.**
                * **Pruned core tags for this waifu are `{", ".join(meta_info["core_tags"])}`.** You can add them to the prompt when some features of waifu (e.g. hair color) are not stable.
                * The base model used for training is [{train_pretrained_model}](https://huggingface.co/{train_pretrained_model}). Architecture is `{base_model_type}`.
                * Dataset used for training is the `{dataset_info["name"]}` in [{dataset_info["repository"]}](https://huggingface.co/datasets/{dataset_info["repository"]}), which contains {plural_word(dataset_info["size"], "image")}.
                * Batch size is {meta_info["train"]["dataset"]["bs"]}, resolution is {ds_res}x{ds_res}, clustering into {plural_word(meta_info["train"]["dataset"]["num_bucket"], "bucket")}.
                * Batch size for regularization dataset is {meta_info["train"]["reg_dataset"]["bs"]}, resolution is {reg_res}x{reg_res}, clustering into {plural_word(meta_info["train"]["reg_dataset"]["num_bucket"], "bucket")}.
                * Trained for {plural_word(meta_info["train"]["train"]["train_steps"], "step")}, {plural_word(len(meta_info["steps"]), "checkpoint")} were saved and evaluated.
                * **The step we auto-selected is {step} to balance the fidelity and controllability of the model.**
            """).strip(),
        )
        with open(local_metrics_plot_png, 'rb') as f:
            webhook.add_file(file=f.read(), filename=os.path.basename(local_metrics_plot_png))
        webhook.execute()

        upload_batch_size = 10
        batch = int(math.ceil(len(df) / upload_batch_size))
        for batch_id in range(batch):
            webhook = DiscordWebhook(
                url=os.environ['DC_MODEL_WEBHOOK'],
                content=textwrap.dedent(f"""
                    {plural_word(len(df), 'image')} here for preview.
                """).strip() if batch_id == 0 else "",
            )

            for df_record in df[upload_batch_size * batch_id: upload_batch_size * (batch_id + 1)].to_dict('records'):
                img_file = os.path.join(td, df_record['image'])
                with open(img_file, 'rb') as f:
                    webhook.add_file(file=f.read(), filename=os.path.basename(img_file))

            webhook.execute()

        # can not upload large files
        webhook = DiscordWebhook(
            url=os.environ['DC_MODEL_WEBHOOK'],
            content=f'{"LyCORIS" if is_lycoris else "LoRA"} Model {"files" if len(model_files) > 1 else "file"} of `{meta_info["display_name"]}`'
        )
        for model_file in model_files:
            with open(model_file, 'rb') as f:
                webhook.add_file(file=f.read(), filename=os.path.basename(model_file))
        response = webhook.execute()
        response.raise_for_status()
