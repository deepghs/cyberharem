import datetime
import os
import pathlib

from ditk import logging
from hbutils.system import TemporaryDirectory
from huggingface_hub import CommitOperationAdd
from huggingface_hub.utils import RepositoryNotFoundError

from .export import export_workdir
from .steps import find_steps_in_workdir
from ..utils import get_hf_client


def deploy_to_huggingface(workdir: str, repository=None, revision: str = 'main', n_repeats: int = 3):
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
