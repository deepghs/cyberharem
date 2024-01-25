import os
from functools import partial

import click
from ditk import logging
from gchar.generic import import_generic
from gchar.utils import GLOBAL_CONTEXT_SETTINGS
from gchar.utils import print_version as _origin_print_version
from hbutils.system import TemporaryDirectory
from huggingface_hub import hf_hub_url
from tqdm.auto import tqdm

from cyberharem.dataset import save_recommended_tags
from cyberharem.publish import find_steps_in_workdir
from cyberharem.utils import get_hf_fs, download_file
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
@click.option('-C', '--clip_skip', 'clip_skip', type=int, default=2,
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


@cli.command('rehf', context_settings={**GLOBAL_CONTEXT_SETTINGS}, help='Re-Publish to huggingface')
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
@click.option('-C', '--clip_skip', 'clip_skip', type=int, default=2,
              help='Clip skip.', show_default=True)
@click.option('-S', '--infer_steps', 'infer_steps', type=int, default=30,
              help='Steps of inference.', show_default=True)
def rehf(repository, revision, n_repeats, pretrained_model,
         width, height, clip_skip, infer_steps):
    logging.try_init_root(logging.INFO)
    with TemporaryDirectory() as workdir:
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
@click.option('--allow_nsfw', '-N', 'allow_nsfw', is_flag=True, type=bool, default=False,
              help='Allow uploading nsfw images.', show_default=True)
@click.option('--version_name', '-v', 'version_name', type=str, default=None,
              help='Name of the version.', show_default=True)
@click.option('--force_create', '-F', 'force_create', is_flag=True, type=bool, default=False,
              help='Force create new model.', show_default=True)
@click.option('--no_ccip', 'no_ccip_check', is_flag=True, type=bool, default=False,
              help='No CCIP check.', show_default=True)
def civitai(repository, title, steps, epochs, draft, publish_time, allow_nsfw,
            version_name, force_create, no_ccip_check):
    logging.try_init_root(logging.INFO)
    model_id = civitai_publish_from_hf(
        repository, title,
        step=steps, epoch=epochs, draft=draft,
        publish_at=publish_time, allow_nsfw_images=allow_nsfw,
        version_name=version_name, force_create_model=force_create,
        no_ccip_check=no_ccip_check,
    )
    url = f'https://civitai.com/models/{model_id}'
    if not draft:
        logging.info(f'Deploy success, model now can be seen at {url} .')
    else:
        logging.info(f'Draft created, it can be seed at {url} .')


if __name__ == '__main__':
    cli()
