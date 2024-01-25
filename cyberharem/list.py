import fnmatch
from functools import partial

import click
from gchar.generic import import_generic
from gchar.utils import GLOBAL_CONTEXT_SETTINGS
from gchar.utils import print_version as _origin_print_version

from cyberharem.utils import get_hf_client, get_global_namespace

print_version = partial(_origin_print_version, 'cyberharem.train')

import_generic()


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Publish trained models')
@click.option('-v', '--version', is_flag=True, callback=print_version, expose_value=False, is_eager=True)
def cli():
    pass  # pragma: no cover


@cli.command('models', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='List models')
@click.option('-p', '--pattern', 'pattern', type=str, default='*',
              help='Pattern of models.', show_default=True)
def models(pattern):
    hf_client = get_hf_client()
    for model in hf_client.list_models(author=get_global_namespace()):
        if fnmatch.fnmatch(model.modelId, pattern):
            print(model.modelId)


@cli.command('datasets', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='List datasets')
@click.option('-p', '--pattern', 'pattern', type=str, default='*',
              help='Pattern of models.', show_default=True)
def datasets(pattern):
    hf_client = get_hf_client()
    for ds in hf_client.list_datasets(author=get_global_namespace()):
        if fnmatch.fnmatch(ds.id, pattern):
            print(ds.id)


if __name__ == '__main__':
    cli()
