import re
from functools import partial

import click
from ditk import logging
from gchar.generic import import_generic
from gchar.utils import GLOBAL_CONTEXT_SETTINGS
from gchar.utils import print_version as _origin_print_version
from unidecode import unidecode

from .bangumibase import sync_bangumi_base
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
def huggingface(video_or_directory: str, bangumi_name: str,
                repository: str, revision: str = 'main', min_size: int = 320, no_extract: bool = False):
    logging.try_init_root(logging.INFO)
    rname = re.sub(r'[\W_]+', '', unidecode(bangumi_name.lower()))
    repository = repository or f"{get_global_bg_namespace()}/{rname}"
    extract_to_huggingface(
        video_or_directory, bangumi_name, repository, revision,
        no_extract=no_extract, min_size=min_size
    )


@cli.command('bgsync', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help=f'Sync index on {get_global_bg_namespace()}')
@click.option('--repository', '-r', 'repository', type=str, default=f'{get_global_bg_namespace()}/README',
              help='Repository to publish to.', show_default=True)
def bgsync(repository: str):
    logging.try_init_root(logging.INFO)
    sync_bangumi_base(repository)


if __name__ == '__main__':
    cli()
