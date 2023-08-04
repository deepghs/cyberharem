import os.path
from functools import partial

import click
from ditk import logging
from gchar.utils import GLOBAL_CONTEXT_SETTINGS
from gchar.utils import print_version as _origin_print_version
from huggingface_hub import hf_hub_url
from tqdm.auto import tqdm

from ..utils import get_hf_fs, download_file

print_version = partial(_origin_print_version, 'cyberharem.train')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Publish trained models')
@click.option('-v', '--version', is_flag=True, callback=print_version, expose_value=False, is_eager=True)
def cli():
    pass  # pragma: no cover


@cli.command('download', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Download trained ckpts from huggingface.')
@click.option('-r', '--repository', 'repository', type=str, required=True,
              help='Repository.', show_default=True)
@click.option('-w', '--workdir', 'workdir', type=str, required=True,
              help='Work directory', show_default=True)
def download(repository, workdir):
    logging.try_init_root(logging.INFO)

    hf_fs = get_hf_fs()
    for f in tqdm(hf_fs.glob(f'{repository}/*/previews/*')):
        rel_file = os.path.relpath(f, repository)
        local_file = os.path.join(workdir, 'ckpts', rel_file)
        if os.path.dirname(local_file):
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
        download_file(
            hf_hub_url(repository, filename=rel_file),
            local_file
        )


if __name__ == '__main__':
    cli()
