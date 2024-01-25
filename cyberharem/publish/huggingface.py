import datetime
import glob
import json
import math
import os
import pathlib
import shutil
import time
import zipfile
from textwrap import dedent
from typing import Optional

import pandas as pd
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, hf_hub_url
from huggingface_hub.utils import RepositoryNotFoundError
from imgutils.sd import get_sdmeta_from_image
from tqdm import tqdm

from .convert import convert_to_webui_lora, pack_to_bundle_lora
from .export import export_workdir, _GITLFS, EXPORT_MARK
from .steps import find_steps_in_workdir
from ..eval import eval_for_workdir, list_steps
from ..infer.draw import _DEFAULT_INFER_MODEL, list_rtag_names
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


def deploy_to_hf(workdir: str, repository: Optional[str] = None, eval_cfgs: Optional[dict] = None,
                 steps_batch_size: int = 5):
    eval_dir = os.path.join(workdir, 'eval')
    if os.path.join(os.path.join(eval_dir, 'metrics_selected.csv')):
        logging.info(f'Completed evaluation detected on {eval_dir!r}.')
    else:
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

    def _make_table_for_steps(stps, full: bool = False):
        if full:
            columns = ['Step', 'Epoch', 'CCIP', 'AI Corrupt', 'Bikini Plus', 'Score', 'Download', *rtag_names]
        else:
            columns = ['Step', 'Epoch', 'Score', 'Download', *rtag_names]

        data = []
        for s in stps:
            download_url = hf_hub_url(
                repo_id=repository, repo_type="model",
                filename=os.path.join(str(s), f"{name}.zip"),
            )
            if full:
                row = [
                    s,
                    infos[s]["epoch"],
                    f'{infos[s]["ccip"]:.3f}',
                    f'{infos[s]["aic"]:.3f}',
                    f'{infos[s]["bp"]:.3f}',
                    f'{infos[s]["integrate"]:.3f}',
                    f'[Download]({download_url})',
                    *(f'![{rt}]({s}/previews/{rt}.png)' for rt in rtag_names)
                ]
            else:
                row = [
                    s,
                    infos[s]["epoch"],
                    f'{infos[s]["integrate"]:.3f}',
                    f'[Download]({download_url})',
                    *(f'![{rt}]({s}/previews/{rt}.png)' for rt in rtag_names)
                ]
            data.append(row)

        return pd.DataFrame(data, columns=columns)

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

                ds_res = meta_info["train"]["dataset"]["resolution"]
                print(f'* Batch size is {meta_info["train"]["dataset"]["bs"]}, '
                      f'resolution is {ds_res}x{ds_res}, '
                      f'clustering into {plural_word(meta_info["train"]["dataset"]["num_bucket"], "bucket")}.', file=f)

                reg_res = meta_info["train"]["reg_dataset"]["resolution"]
                print(f'* Batch size for regularization dataset is {meta_info["train"]["reg_dataset"]["bs"]}, '
                      f'resolution is {reg_res}x{reg_res}, '
                      f'clustering into {plural_word(meta_info["train"]["reg_dataset"]["num_bucket"], "bucket")}.',
                      file=f)
                print(f'* Trained for {plural_word(meta_info["train"]["train"]["train_steps"], "step")}.', file=f)
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
                print(f'![Metrics Plot](metrics_plot.png)', file=f)
                print(f'', file=f)

                sample_meta = get_sdmeta_from_image(glob.glob(os.path.join(workdir, 'eval', '*', '*.png'))[0])
                infer_pretrained_model = sample_meta.parameters['Model']
                print(f'The base model used for generating preview images is '
                      f'[{infer_pretrained_model}](https://huggingface.co/{infer_pretrained_model}).', file=f)
                print(f'', file=f)

                print(f'Here are the preview of the recommended steps', file=f)
                print(f'', file=f)
                print(_make_table_for_steps(selected_steps, full=False).to_markdown(index=False), file=f)
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
                print(f'We list table of all steps, as the following', file=f)
                f_table = _make_table_for_steps(steps[::-1], full=True)
                batch_count = int(math.ceil(len(f_table) / steps_batch_size))
                for i in range(batch_count):
                    s_table = f_table[i * steps_batch_size: (i + 1) * steps_batch_size]
                    print(f'<details>', file=f)
                    print(f'<summary>Steps From {s_table["Step"].min()} to {s_table["Step"].max()}</summary>', file=f)
                    print(f'', file=f)
                    print(s_table.to_markdown(index=False), file=f)
                    print(f'</details>', file=f)
                    print(f'', file=f)

        logging.info(f'Uploading files to repository {repository!r} ...')
        upload_directory_as_directory(
            local_directory=td,
            path_in_repo='.',
            repo_id=repository,
            repo_type='model',
            message=f'Upload model for {meta_info["display_name"]}',
            clear=True,
        )


def deploy_to_huggingface(workdir: str, repository=None, revision: str = 'main', n_repeats: int = 3,
                          pretrained_model: str = _DEFAULT_INFER_MODEL, clip_skip: int = 2,
                          image_width: int = 512, image_height: int = 768, infer_steps: int = 30,
                          lora_alpha: float = 0.85, sample_method: str = 'DPM++ 2M Karras',
                          model_hash: Optional[str] = None):
    name, _ = find_steps_in_workdir(workdir)
    repository = repository or f'CyberHarem/{name}'

    logging.info(f'Initializing repository {repository!r} ...')
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()
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
            commit_message = f'Update {name}\'s .gitattributes, on {current_time}'
            logging.info(f'Updating {name}\'s .gitattributes to repository {repository!r} ...')
            hf_client.create_commit(
                repository,
                operations,
                commit_message=commit_message,
                repo_type='model',
                revision=revision,
            )

    with TemporaryDirectory() as td:
        export_workdir(
            workdir, td, n_repeats, pretrained_model,
            clip_skip, image_width, image_height, infer_steps,
            lora_alpha, sample_method, model_hash, repository,
        )

        try:
            hf_client.repo_info(repo_id=repository, repo_type='dataset')
        except RepositoryNotFoundError:
            has_dataset_repo = False
        else:
            has_dataset_repo = True

        readme_text = pathlib.Path(os.path.join(td, 'README.md')).read_text(encoding='utf-8')
        with open(os.path.join(td, 'README.md'), 'w', encoding='utf-8') as f:
            print('---', file=f)
            print('license: mit', file=f)
            if has_dataset_repo:
                print('datasets:', file=f)
                print(f'- {repository}', file=f)
            print('pipeline_tag: text-to-image', file=f)
            print('tags:', file=f)
            print('- art', file=f)
            print('---', file=f)
            print(f'', file=f)
            print(readme_text, file=f)

        _exist_files = [os.path.relpath(file, repository) for file in hf_fs.glob(f'{repository}/**')]
        _exist_ps = sorted([(file, file.split('/')) for file in _exist_files], key=lambda x: x[1])
        pre_exist_files = set()
        for i, (file, segments) in enumerate(_exist_ps):
            if i < len(_exist_ps) - 1 and segments == _exist_ps[i + 1][1][:len(segments)]:
                continue
            if file != '.':
                pre_exist_files.add(file)

        operations = []
        for directory, _, files in os.walk(td):
            for file in files:
                filename = os.path.abspath(os.path.join(td, directory, file))
                file_in_repo = os.path.relpath(filename, td)
                operations.append(CommitOperationAdd(
                    path_in_repo=file_in_repo,
                    path_or_fileobj=filename,
                ))
                if file_in_repo in pre_exist_files:
                    pre_exist_files.remove(file_in_repo)
        logging.info(f'Useless files: {sorted(pre_exist_files)} ...')
        for file in sorted(pre_exist_files):
            operations.append(CommitOperationDelete(path_in_repo=file))

        current_time = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
        commit_message = f'Publish {name}\'s lora, on {current_time}'
        logging.info(f'Publishing {name}\'s lora to repository {repository!r} ...')
        hf_client.create_commit(
            repository,
            operations,
            commit_message=commit_message,
            repo_type='model',
            revision=revision,
        )
