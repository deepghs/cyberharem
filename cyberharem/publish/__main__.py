from functools import partial

import click
from ditk import logging
from gchar.generic import import_generic
from gchar.utils import GLOBAL_CONTEXT_SETTINGS
from gchar.utils import print_version as _origin_print_version

from .civitai import civitai_publish_from_hf
from .huggingface import deploy_to_huggingface
from ..infer.draw import _DEFAULT_INFER_MODEL

import_generic()

print_version = partial(_origin_print_version, 'cyberharem')


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Publish trained models')
@click.option('-v', '--version', is_flag=True, callback=print_version, expose_value=False, is_eager=True)
def cli():
    pass  # pragma: no cover


@cli.command('huggingface', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Publish to huggingface')
@click.option('-w', '--workdir', 'workdir', type=click.Path(file_okay=False, exists=True), required=True,
              help='Work directory for experiment.', show_default=True)
@click.option('--repository', '-r', 'repository', type=str, default=None,
              help='Repository to publish to.', show_default=True)
@click.option('--revision', '-R', 'revision', type=str, default='main',
              help='Revision for pushing the model.', show_default=True)
@click.option('-n', '--n_repeats', 'n_repeats', type=int, default=3,
              help='N Repeats for text encoder', show_default=True)
@click.option('-m', '--pretrained_model', 'pretrained_model', type=str, default=_DEFAULT_INFER_MODEL,
              help='Pretrained model for preview drawing.', show_default=True)
@click.option('--width', 'width', type=int, default=512,
              help='Width of images.', show_default=True)
@click.option('--height', 'height', type=int, default=768,
              help='Height of images.', show_default=True)
@click.option('-C', '--clip_skip', 'clip_skip', type=int, default=1,
              help='Clip skip.', show_default=True)
@click.option('-S', '--infer_steps', 'infer_steps', type=int, default=30,
              help='Steps of inference.', show_default=True)
def huggingface(workdir: str, repository, revision, n_repeats, pretrained_model,
                width, height, clip_skip, infer_steps):
    logging.try_init_root(logging.INFO)
    deploy_to_huggingface(
        workdir, repository, revision, n_repeats, pretrained_model,
        clip_skip, width, height, infer_steps,
    )


@cli.command('civitai', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Publish to huggingface')
@click.option('--repository', '-r', 'repository', type=str, required=True,
              help='Repository to publish from.', show_default=True)
@click.option('--title', '-t', 'title', type=str, default=None,
              help='Title of the civitai model.', show_default=True)
@click.option('--steps', '-s', 'steps', type=int, default=None,
              help='Steps to deploy.', show_default=True)
@click.option('--epochs', '-e', 'epochs', type=int, default=None,
              help='Epochs to deploy.', show_default=True)
@click.option('--draft', '-d', 'draft', is_flag=True, type=bool, default=False,
              help='Only create draft without publishing.', show_default=True)
@click.option('--time', '-T', 'publish_time', type=str, default=None,
              help='Publish time, publish immediately when not given.', show_default=True)
@click.option('--safe_only', '-S', 'safe_only', is_flag=True, type=bool, default=False,
              help='Upload safe images only.', show_default=True)
def civitai(repository, title, steps, epochs, draft, publish_time, safe_only):
    logging.try_init_root(logging.INFO)
    model_id = civitai_publish_from_hf(
        repository, title,
        step=steps, epoch=epochs, draft=draft,
        publish_at=publish_time, safe_only=safe_only,
    )
    url = f'https://civitai.com/models/{model_id}'
    if not draft:
        logging.info(f'Deploy success, model now can be seen at {url} .')
    else:
        logging.info(f'Draft created, it can be seed at {url} .')


if __name__ == '__main__':
    cli()
