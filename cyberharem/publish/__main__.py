from functools import partial

import click
from ditk import logging
from gchar.generic import import_generic
from gchar.utils import GLOBAL_CONTEXT_SETTINGS
from gchar.utils import print_version as _origin_print_version

from .civitai import civitai_upload_from_hf
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
@click.option('-m', '--pretrained_model', 'pretrained_model', type=str, default=_DEFAULT_INFER_MODEL,
              help='Pretrained model for preview drawing.', show_default=True)
@click.option('--width', 'width', type=int, default=832,
              help='Width of images.', show_default=True)
@click.option('--height', 'height', type=int, default=1216,
              help='Height of images.', show_default=True)
@click.option('-C', '--clip_skip', 'clip_skip', type=int, default=2,
              help='Clip skip.', show_default=True)
@click.option('-S', '--infer_steps', 'infer_steps', type=int, default=30,
              help='Steps of inference.', show_default=True)
def huggingface(workdir: str, repository, pretrained_model, width, height, clip_skip, infer_steps):
    logging.try_init_root(logging.INFO)
    deploy_to_huggingface(
        workdir=workdir,
        repository=repository,
        eval_cfgs={
            'pretrained_model': pretrained_model,
            'clip_skip': clip_skip,
            'infer_steps': infer_steps,
            'width': width,
            'height': height,
        },
    )


@cli.command('civitai', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Publish to huggingface')
@click.option('--repository', '-r', 'repository', type=str, required=True,
              help='Repository to publish from.', show_default=True)
@click.option('--step', '-s', 'step', type=int, default=None,
              help='Step to deploy.', show_default=True)
@click.option('--draft', '-d', 'draft', is_flag=True, type=bool, default=False,
              help='Only create draft without publishing.', show_default=True)
@click.option('--time', '-T', 'publish_time', type=str, default=None,
              help='Publish time, publish immediately when not given.', show_default=True)
@click.option('--allow_nsfw', '-N', 'allow_nsfw', is_flag=True, type=bool, default=False,
              help='Allow uploading nsfw images.', show_default=True)
@click.option('--update_description_only', '-D', 'update_description_only', is_flag=True, type=bool, default=False,
              help='Update the model description only.', show_default=True)
@click.option('--existing_model_id', 'existing_model_id', type=int, default=None,
              help='Existing model ID', show_default=True)
def civitai(repository, step, draft, publish_time, allow_nsfw, update_description_only, existing_model_id):
    logging.try_init_root(logging.INFO)
    civitai_upload_from_hf(
        repository=repository,
        step=step,
        allow_nsfw=allow_nsfw,
        publish_at=publish_time,
        draft=draft,
        update_description_only=update_description_only,
        existing_model_id=existing_model_id,
    )


if __name__ == '__main__':
    cli()
