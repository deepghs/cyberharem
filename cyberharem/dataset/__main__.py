from functools import partial
from typing import Optional

import click
from ditk import logging
from gchar.utils import GLOBAL_CONTEXT_SETTINGS
from gchar.utils import print_version as _origin_print_version

from .tags import save_recommended_tags

print_version = partial(_origin_print_version, 'cyberharem.dataset')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Publish trained models')
@click.option('-v', '--version', is_flag=True, callback=print_version, expose_value=False, is_eager=True)
def cli():
    pass  # pragma: no cover


@cli.command('retag', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Regenerate tags for given work directory.')
@click.option('-w', '--workdir', 'workdir', type=click.Path(file_okay=False, exists=True), required=True,
              help='Work directory for experiment.', show_default=True)
@click.option('-n', '--name', 'name', type=str, default=None,
              help='Name of the character.', show_default=True)
def retag(workdir, name: Optional[str] = None):
    logging.try_init_root(logging.INFO)

    from ..publish.steps import find_steps_in_workdir
    pt_name, _ = find_steps_in_workdir(workdir)
    name = name or '_'.join(pt_name.split('_')[:-1])

    logging.info(f'Regenerate tags for {name!r}, on {workdir!r}.')
    save_recommended_tags(name, workdir=workdir)
    logging.info('Success!')


if __name__ == '__main__':
    cli()
