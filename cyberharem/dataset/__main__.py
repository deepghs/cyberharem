from functools import partial

import click
from ditk import logging
from gchar.utils import GLOBAL_CONTEXT_SETTINGS
from gchar.utils import print_version as _origin_print_version

from .discord import publish_to_discord

print_version = partial(_origin_print_version, 'cyberharem.dataset')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Publish dataset')
@click.option('-v', '--version', is_flag=True, callback=print_version, expose_value=False, is_eager=True)
def cli():
    pass  # pragma: no cover


@cli.command('hf_discord', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Publish to discord from huggingface')
@click.option('--repository', '-r', 'repository', type=str, default=None,
              help='Repository to publish to.', show_default=True)
def hf_discord(repository: str):
    logging.try_init_root(logging.INFO)
    publish_to_discord(repository)


if __name__ == '__main__':
    cli()
