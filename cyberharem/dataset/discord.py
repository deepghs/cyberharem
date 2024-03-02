import glob
import io
import json
import os
import textwrap

from discord_webhook import DiscordWebhook
from hbutils.scale import size_to_bytes_str
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import download_archive_as_directory
from huggingface_hub import hf_hub_url
from waifuc.action import AlignMaxAreaAction, RatingFilterAction
from waifuc.source import LocalSource

from ..utils import get_hf_fs
from ..utils.ghaction import GithubActionClient


def send_discord_publish_to_github_action(repository: str):
    client = GithubActionClient()
    client.create_workflow_run(
        'deepghs/cyberharem',
        'DC Dataset Publish',
        data={
            'repository': repository,
        }
    )


def publish_to_discord(repository: str):
    hf = get_hf_fs()
    meta_info = json.loads(hf.read_text(f'datasets/{repository}/meta.json'))

    hf_url = f'https://huggingface.co/datasets/{repository}'
    with io.StringIO() as sio:
        print(textwrap.dedent(f"""
            Dataset of `{meta_info['display_name']}` has been published to huggingface repository: {hf_url}.
            * The trigger word expected is `{meta_info['name']}`.
            * Core tags of this character: `{', '.join(meta_info['core_tags'])}`. **They are pruned in this dataset.**
            * **This dataset contains {plural_word(meta_info['base_size'], 'image')} as base**.
            * {plural_word(len(meta_info['clusters']), 'cluster')} detected in this dataset.
            * The `raw` dataset are not pruned or pre-processed. You can directly use it in waifuc.
        """).strip(), file=sio)

        for key, data in meta_info['packages'].items():
            print(
                f'* Package `{key}` contains {plural_word(data["size"], "image")}. '
                f'Type: `{data["type"]}`, Size: `{size_to_bytes_str(data["package_size"], precision=2)}`, '
                f'[Download Link]({hf_hub_url(repo_id=repository, filename=data["filename"], repo_type="dataset")}).',
                file=sio
            )

        webhook = DiscordWebhook(
            url=os.environ['DC_DATASET_WEBHOOK'],
            content=sio.getvalue(),
        )
        with TemporaryDirectory() as td:
            origin_dir = os.path.join(td, 'source')
            os.makedirs(origin_dir, exist_ok=True)
            download_archive_as_directory(
                repo_id=repository,
                file_in_repo='dataset-raw.zip',
                local_directory=origin_dir,
                repo_type='dataset',
            )

            dst_dir = os.path.join(td, 'dst')
            os.makedirs(dst_dir, exist_ok=True)
            LocalSource(origin_dir, shuffle=True).attach(
                AlignMaxAreaAction(1200),
                RatingFilterAction(['safe', 'r15']),
            )[:10].export(dst_dir)

            for img_file in glob.glob(os.path.join(dst_dir, '*.png')):
                with open(img_file, 'rb') as f:
                    webhook.add_file(file=f.read(), filename=os.path.basename(img_file))

        webhook.execute()
