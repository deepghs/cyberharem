import datetime
import os
import pathlib
from typing import Optional

from ditk import logging
from hbutils.system import TemporaryDirectory
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub.utils import RepositoryNotFoundError

from .export import export_workdir, _GITLFS
from .steps import find_steps_in_workdir
from ..infer.draw import _DEFAULT_INFER_MODEL
from ..utils import get_hf_client, get_hf_fs


def deploy_to_huggingface(workdir: str, repository=None, revision: str = 'main', n_repeats: int = 3,
                          pretrained_model: str = _DEFAULT_INFER_MODEL, clip_skip: int = 2,
                          image_width: int = 512, image_height: int = 768, infer_steps: int = 30,
                          lora_alpha: float = 0.85, sample_method: str = 'DPM++ 2M Karras',
                          model_hash: Optional[str] = None):
    name, _ = find_steps_in_workdir(workdir)
    repository = repository or f'CyberHarem/{name}'

    logging.info(f'Initializing repository {repository!r} ...')
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()
    if not hf_fs.exists(f'{repository}/.gitattributes'):
        hf_client.create_repo(repo_id=repository, repo_type='model', exist_ok=True)

    if not hf_fs.exists(f'{repository}/.gitattributes') or \
            '*.png filter=lfs diff=lfs merge=lfs -text' not in hf_fs.read_text(f'{repository}/.gitattributes'):
        logging.info(f'Preparing for lfs attributes of repository {repository!r}.')
        with TemporaryDirectory() as td:
            _git_attr_file = os.path.join(td, '.gitattributes')
            with open(_git_attr_file, 'w', encoding='utf-8') as f:
                print(_GITLFS, file=f)

            operations = [
                CommitOperationAdd(
                    path_in_repo='.gitattributes',
                    path_or_fileobj=_git_attr_file,
                )
            ]

            current_time = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
            commit_message = f'Update {name}\'s .gitattributes, on {current_time}'
            logging.info(f'Updating {name}\'s .gitattributes to repository {repository!r} ...')
            hf_client.create_commit(
                repository,
                operations,
                commit_message=commit_message,
                repo_type='model',
                revision=revision,
            )

    with TemporaryDirectory() as td:
        export_workdir(
            workdir, td, n_repeats, pretrained_model,
            clip_skip, image_width, image_height, infer_steps,
            lora_alpha, sample_method, model_hash, repository,
        )

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

        _exist_files = [os.path.relpath(file, repository) for file in hf_fs.glob(f'{repository}/**')]
        _exist_ps = sorted([(file, file.split('/')) for file in _exist_files], key=lambda x: x[1])
        pre_exist_files = set()
        for i, (file, segments) in enumerate(_exist_ps):
            if i < len(_exist_ps) - 1 and segments == _exist_ps[i + 1][1][:len(segments)]:
                continue
            if file != '.':
                pre_exist_files.add(file)

        operations = []
        for directory, _, files in os.walk(td):
            for file in files:
                filename = os.path.abspath(os.path.join(td, directory, file))
                file_in_repo = os.path.relpath(filename, td)
                operations.append(CommitOperationAdd(
                    path_in_repo=file_in_repo,
                    path_or_fileobj=filename,
                ))
                if file_in_repo in pre_exist_files:
                    pre_exist_files.remove(file_in_repo)
        logging.info(f'Useless files: {sorted(pre_exist_files)} ...')
        for file in sorted(pre_exist_files):
            operations.append(CommitOperationDelete(path_in_repo=file))

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
