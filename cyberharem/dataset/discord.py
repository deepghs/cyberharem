import io
import json
import os
import textwrap

from discord_webhook import DiscordWebhook
from hbutils.scale import size_to_bytes_str
from hbutils.string import plural_word
from huggingface_hub import hf_hub_url

from cyberharem.utils import get_hf_fs


def publish_to_discord(repository: str):
    hf = get_hf_fs()
    meta_info = json.loads(hf.read_text(f'datasets/{repository}/meta.json'))

    hf_url = f'https://huggingface.co/datasets/{repository}'
    with io.StringIO()as sio:
        print(textwrap.dedent(f"""
            Dataset of `{meta_info['display_name']}` has been published to huggingface repository: {hf_url}.
            * The trigger word expected is `{meta_info['name']}`.
            * Core tags of this model: `{', '.join(meta_info['core_tags'])}`.
            * This dataset contains {plural_word(meta_info['base_size'], 'image')} as base.
            * {plural_word(len(meta_info['clusters']), 'cluster')} detected in this dataset.
        """).strip(), file=sio)

        for key, data in meta_info['packages'].items():
            print(f'* Package `{key}` contains {plural_word(data["size"], "image")}. '
                  f'Type: `{data["type"]}`, Size: `{size_to_bytes_str(data["package_size"], precision=2)}`, '
                  f'[Download Link]({hf_hub_url(repo_id=repository, filename=data["filename"], repo_type="dataset")}), '
                  f'Description: {data["description"]}.', file=sio)

        webhook = DiscordWebhook(
            url=os.environ['DC_DATASET_WEBHOOK'],
            content=sio.getvalue(),
        )
        webhook.execute()
