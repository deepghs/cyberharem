import json
import math
import os
import textwrap
from typing import Optional

import numpy as np
import pandas as pd
from discord_webhook import DiscordWebhook
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import download_archive_as_directory, download_file_to_file
from huggingface_hub import hf_hub_url
from imgutils.validate import anime_rating
from tqdm import tqdm

from ..utils import get_hf_fs
from ..utils.ghaction import GithubActionClient


def send_discord_publish_to_github_action(repository: str):
    client = GithubActionClient()
    client.create_workflow_run(
        os.environ.get('GITHUB_REPOSITORY') or 'deepghs/cyberharem',
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
        pass
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
        train_toml_url = hf_hub_url(repo_id=repository, repo_type='model', filename=f'train.toml')
        webhook = DiscordWebhook(
            url=os.environ['DC_MODEL_WEBHOOK'],
            content=textwrap.dedent(f"""
                LoRA Model of `{meta_info['display_name']}` has been published to huggingface repository: {hf_url}.
                * **Trigger word is `{name}`.**
                * **Pruned core tags for this waifu are `{", ".join(meta_info["core_tags"])}`.** You can add them to the prompt when some features of waifu (e.g. hair color) are not stable.
                * The base model used for training is [{train_pretrained_model}](https://huggingface.co/{train_pretrained_model}). Architecture is `{base_model_type}`.
                * Dataset used for training is the `{dataset_info["name"]}` in [{dataset_info["repository"]}](https://huggingface.co/datasets/{dataset_info["repository"]}), which contains {plural_word(dataset_info["size"], "image")}.
                * For more details in training, you can take a look at [training configuration file]({train_toml_url}).
                * For more details in LoRA, you can download it, and read the metadata with a1111\'s webui.
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
            content=f'LoRA Model {"files" if len(model_files) > 1 else "file"} of `{meta_info["display_name"]}`'
        )
        for model_file in model_files:
            with open(model_file, 'rb') as f:
                webhook.add_file(file=f.read(), filename=os.path.basename(model_file))
        response = webhook.execute()
        response.raise_for_status()
