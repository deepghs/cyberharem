import os.path
from functools import partial

import click
from ditk import logging
from gchar.generic import import_generic
from gchar.utils import GLOBAL_CONTEXT_SETTINGS
from gchar.utils import print_version as _origin_print_version
from huggingface_hub import hf_hub_url
from tqdm.auto import tqdm

from cyberharem.dataset import save_recommended_tags
from cyberharem.publish import find_steps_in_workdir
from ..utils import get_hf_fs, download_file

print_version = partial(_origin_print_version, 'cyberharem.train')

import_generic()


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Publish trained models')
@click.option('-v', '--version', is_flag=True, callback=print_version, expose_value=False, is_eager=True)
def cli():
    pass  # pragma: no cover


@cli.command('download', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Download trained ckpts from huggingface.')
@click.option('-r', '--repository', 'repository', type=str, required=True,
              help='Repository.', show_default=True)
@click.option('-w', '--workdir', 'workdir', type=str, default=None,
              help='Work directory', show_default=True)
@click.option('--no-tags', 'no_tags', is_flag=True, type=bool, default=False,
              help='Do not generate tags.', show_default=True)
def download(repository, workdir, no_tags):
    logging.try_init_root(logging.INFO)
    workdir = workdir or os.path.join('runs', repository.split('/')[-1])

    logging.info(f'Downloading models for {workdir!r} ...')
    hf_fs = get_hf_fs()
    for f in tqdm(hf_fs.glob(f'{repository}/*/raw/*')):
        rel_file = os.path.relpath(f, repository)
        local_file = os.path.join(workdir, 'ckpts', os.path.basename(rel_file))
        if os.path.dirname(local_file):
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
        download_file(
            hf_hub_url(repository, filename=rel_file),
            local_file
        )

    if not no_tags:
        logging.info(f'Regenerating tags for {workdir!r} ...')
        pt_name, _ = find_steps_in_workdir(workdir)
        game_name = pt_name.split('_')[-1]
        name = '_'.join(pt_name.split('_')[:-1])

        from gchar.games.dispatch.access import GAME_CHARS
        if game_name in GAME_CHARS:
            ch_cls = GAME_CHARS[game_name]
            ch = ch_cls.get(name)
        else:
            ch = None

        if ch is None:
            source = repository
        else:
            source = ch

        logging.info(f'Regenerate tags for {source!r}, on {workdir!r}.')
        save_recommended_tags(source, name=pt_name, workdir=workdir)
        logging.info('Success!')


if __name__ == '__main__':
    cli()
