import datetime
import os
import pathlib
from functools import partial

import click
from ditk import logging
from gchar.utils import GLOBAL_CONTEXT_SETTINGS
from gchar.utils import print_version as _origin_print_version
from hbutils.system import TemporaryDirectory
from huggingface_hub import CommitOperationAdd
from huggingface_hub.utils import RepositoryNotFoundError

from .export import export_workdir
from .steps import find_steps_in_workdir
from ..utils import get_hf_client

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
    name, _ = find_steps_in_workdir(workdir)
    repository = repository or f'CyberHarem/{name}'

    logging.info(f'Initializing repository {repository!r} ...')
    hf_client = get_hf_client()
    hf_client.create_repo(repo_id=repository, repo_type='model', exist_ok=True)

    with TemporaryDirectory() as td:
        export_workdir(workdir, td, n_repeats)

        try:
            hf_client.repo_info(repo_id=repository, repo_type='dataset')
        except RepositoryNotFoundError:
            has_dataset_repo = False
        else:
            has_dataset_repo = True

        readme_text = pathlib.Path(os.path.join(td, 'README.md')).read_text(encoding='utf-8')
        with open(os.path.join(td, 'README.md'), 'w', encoding='utf-8') as f:
            print('---', file=f)
            print('license: mit', file=f)
            if has_dataset_repo:
                print('datasets:', file=f)
                print(f'- {repository}', file=f)
            print('pipeline_tag: text-to-image', file=f)
            print('tags:', file=f)
            print('- art', file=f)
            print('---', file=f)
            print('', file=f)
            print(readme_text, file=f)

        operations = []
        for directory, _, files in os.walk(td):
            for file in files:
                filename = os.path.abspath(os.path.join(td, directory, file))
                file_in_repo = os.path.relpath(filename, td)
                operations.append(CommitOperationAdd(
                    path_in_repo=file_in_repo,
                    path_or_fileobj=filename,
                ))

        current_time = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
        commit_message = f'Publish {name}\'s lora, on {current_time}'
        logging.info(f'Publishing {name}\'s lora to repository {repository!r} ...')
        hf_client.create_commit(
            repository,
            operations,
            commit_message=commit_message,
            repo_type='model',
            revision=revision,
        )


if __name__ == '__main__':
    cli()
