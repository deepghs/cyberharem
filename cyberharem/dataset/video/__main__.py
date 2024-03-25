import re
from functools import partial

import click
from ditk import logging
from gchar.generic import import_generic
from gchar.utils import GLOBAL_CONTEXT_SETTINGS
from gchar.utils import print_version as _origin_print_version
from unidecode import unidecode

from .bangumibase import sync_bangumi_base
from .crawler import crawl_base_to_huggingface
from .discord import publish_to_discord
from .extract import extract_to_huggingface
from ...utils import get_global_bg_namespace

import_generic()

print_version = partial(_origin_print_version, 'cyberharem.dataset.video')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Publish video data')
@click.option('-v', '--version', is_flag=True, callback=print_version, expose_value=False, is_eager=True)
def cli():
    pass  # pragma: no cover


@cli.command('huggingface', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Publish to huggingface')
@click.option('--repository', '-r', 'repository', type=str, default=None,
              help='Repository to publish to.', show_default=True)
@click.option('--revision', '-R', 'revision', type=str, default='main',
              help='Revision for pushing the model.', show_default=True)
@click.option('--input', '-i', 'video_or_directory', type=str, required=True,
              help='Input videos.', show_default=True)
@click.option('--name', '-n', 'bangumi_name', type=str, required=True,
              help='Bangumi name', show_default=True)
@click.option('--min_size', '-s', 'min_size', type=int, default=320,
              help='Min size of image.', show_default=True)
@click.option('--no_extract', '-E', 'no_extract', is_flag=True, type=bool, default=False,
              help='No extraction from videos.', show_default=True)
@click.option('--all_frames', '-A', 'all_frames', is_flag=True, type=bool, default=False,
              help='Extract all frames, not only key frames.', show_default=True)
@click.option('--max_images_limit', 'max_images_limit', type=int, default=50000,
              help='Max images limit, to prevent OOM.', show_default=True)
def huggingface(video_or_directory: str, bangumi_name: str,
                repository: str, revision: str = 'main', min_size: int = 320,
                no_extract: bool = False, max_images_limit: int = 50000, all_frames: bool = False):
    logging.try_init_root(logging.INFO)
    rname = re.sub(r'[\W_]+', '', unidecode(bangumi_name.lower()))
    repository = repository or f"{get_global_bg_namespace()}/{rname}"
    extract_to_huggingface(
        video_or_directory, bangumi_name, repository, revision,
        no_extract=no_extract, min_size=min_size,
        max_images_limit=max_images_limit,
        all_frames=all_frames,
    )


@cli.command('hf_discord', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Publish to discord from huggingface')
@click.option('--repository', '-r', 'repository', type=str, default=None,
              help='Repository to publish to.', show_default=True)
def hf_discord(repository: str):
    logging.try_init_root(logging.INFO)
    publish_to_discord(repository)


@cli.command('extract', context_settings={**GLOBAL_CONTEXT_SETTINGS},
             help='Crawl character dataset to standalone repository')
@click.option('--source_repository', '-R', 'source_repository', type=str, required=True,
              help='Repository of bangumi dataset.', show_default=True)
@click.option('--repository', '-r', 'repository', type=str, default=None,
              help='Repository to publish to.', show_default=True)
@click.option('--ch_id', 'character_ids', type=str, required=True,
              help='Character IDs from bangumi repository (seperated with \',\')', show_default=True)
@click.option('--name', '-N', 'name', type=str, required=True,
              help='Name of the character.', show_default=True)
@click.option('--display_name', type=str, default=None,
              help='Display Name of the character', show_default=True)
@click.option('--limit', '-l', 'limit', type=int, default=1000,
              help='Limit number of dataset.', show_default=True)
@click.option('--no_ccip', 'no_ccip', is_flag=True, type=bool, default=False,
              help='Do not run CCIP on dataset.', show_default=True)
@click.option('--db_tag', 'db_tag', type=str, default=None,
              help='Danbooru tag for that.', show_default=True)
def extract(source_repository, repository, character_ids, name, display_name, limit, no_ccip, db_tag):
    ch_ids = sorted(map(int, filter(bool, map(str.strip, re.split(r'\s*,\s*', character_ids)))))
    crawl_base_to_huggingface(
        source_repository=source_repository,
        repository=repository,
        ch_id=ch_ids,
        name=name,
        display_name=display_name,
        limit=limit,
        standalone_ccip=no_ccip,
        db_tag=db_tag,
    )


@cli.command('bgsync', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help=f'Sync index on {get_global_bg_namespace()}')
@click.option('--repository', '-r', 'repository', type=str, default=f'{get_global_bg_namespace()}/README',
              help='Repository to publish to.', show_default=True)
def bgsync(repository: str):
    logging.try_init_root(logging.INFO)
    sync_bangumi_base(repository)


if __name__ == '__main__':
    cli()
