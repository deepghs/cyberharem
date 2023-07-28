from functools import partial

import click
from ditk import logging
from gchar.utils import GLOBAL_CONTEXT_SETTINGS
from gchar.utils import print_version as _origin_print_version

from .huggingface import deploy_to_huggingface

print_version = partial(_origin_print_version, 'cyberharem')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Publish trained models')
@click.option('-v', '--version', is_flag=True, callback=print_version, expose_value=False, is_eager=True)
def cli():
    pass  # pragma: no cover


@cli.command('huggingface', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Publish to huggingface')
@click.option('-w', '--workdir', 'workdir', type=str, required=True,
              help='Work directory of trained models.')
@click.option('--repository', '-r', 'repository', type=str, default=None,
              help='Repository to publish to.', show_default=True)
@click.option('--revision', '-R', 'revision', type=str, default='main',
              help='Revision for pushing the model.', show_default=True)
@click.option('-n', '--n_repeats', 'n_repeats', type=int, default=2,
              help='N Repeats for text encoder', show_default=True)
def huggingface(workdir: str, repository, revision, n_repeats):
    logging.try_init_root(logging.INFO)
    deploy_to_huggingface(workdir, repository, revision, n_repeats)


if __name__ == '__main__':
    cli()
