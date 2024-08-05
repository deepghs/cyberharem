import logging
import os
import pathlib
import shutil
from contextlib import contextmanager
from typing import List, Dict, Union, Tuple, Optional

from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import download_archive_as_directory

from ..utils import is_image_file, is_txt_file, is_npz_file

_DEFAULT_MULS = {
    'head': 0.75,
}


def datasets_arrange(
        dst_dir: str,
        groups: Dict[Union[Tuple[str, ...], str], Union[str, Tuple[str, List[str]]]],
        group_weights: Optional[Dict[str, float]] = None,
        min_raw_repeats: float = 0.05,
):
    group_weights = {**_DEFAULT_MULS, **dict(group_weights or {})}

    image_counts = {}
    for group_tags, group_info in groups.items():
        if isinstance(group_info, tuple):
            group_src_dir, group_append_tags = group_info
        else:
            group_src_dir = group_info
            group_append_tags = []

        if isinstance(group_tags, str):
            group_tags = (group_tags,)
        else:
            group_tags = tuple(group_tags)
        group_name = ','.join(group_tags)

        logging.info(f'Copying {group_name!r} ...')
        for root, dirs, files in os.walk(group_src_dir):
            for file in files:
                src_file = os.path.abspath(os.path.join(root, file))
                dst_file = os.path.abspath(os.path.join(dst_dir, group_name, os.path.relpath(src_file, group_src_dir)))
                if is_image_file(src_file) or is_txt_file(src_file) or is_npz_file(src_file):
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    shutil.copy(src_file, dst_file)
                    if is_image_file(dst_file):
                        image_counts[group_name] = image_counts.get(group_name, 0) + 1

        logging.info(f'Group {group_name!r} ready, {plural_word(image_counts[group_name], "image")} found.')
        if group_append_tags:
            logging.info(f'Updating caption files for group {group_name!r}, append tags: {group_append_tags!r} ...')
            for root, dirs, files in os.walk(os.path.join(dst_dir, group_name)):
                for file in files:
                    src_file = os.path.abspath(os.path.join(root, file))
                    if is_image_file(src_file):
                        body, _ = os.path.splitext(src_file)
                        src_txt_file = f'{body}.txt'
                        origin_txt = pathlib.Path(src_txt_file).read_text() if os.path.exists(src_txt_file) else ''
                        with open(src_txt_file, 'w') as f:
                            f.write(', '.join([*group_append_tags, origin_txt]))
        else:
            logging.info(f'No caption update for group {group_name!r}.')

    max_count = max(image_counts.values())
    for group_name, image_count in image_counts.items():
        group_tags = group_name.split(',')
        group_times = 1.0
        for tag in group_tags:
            group_times *= group_weights.get(tag, 1.0)

        raw_repeats = group_times * max_count / image_count
        repeats = 0 if raw_repeats < min_raw_repeats else int(max(round(raw_repeats), 1))
        if repeats == 0:
            logging.warning(f'Group {group_name!r} will be ignored due to the low weight.')
            shutil.rmtree(os.path.join(dst_dir, group_name))
        else:
            repeats = max(repeats, 1)
            new_name = f'{repeats}_{group_name}'
            logging.info(f'Group {group_name!r} --> {new_name!r} ({plural_word(image_count, "image")})')
            shutil.move(os.path.join(dst_dir, group_name), os.path.join(dst_dir, new_name))


@contextmanager
def arrange_dataset_from_repo(
        repo_id: str, prefix_tags: List[str] = None,
        dataset_name: str = 'stage3-p480-1200', revision: str = 'main',
        attach_revisions: Optional[List[str]] = None,
        main_group_name: Optional[str] = 'origin',
        group_weights: Optional[Dict[str, float]] = None,
        group_attached_tags: Optional[Dict[str, List[str]]] = None,
):
    attach_revisions = list(attach_revisions or [])
    group_attached_tags = dict(group_attached_tags or {})
    prefix_tags = list(prefix_tags or [])
    with TemporaryDirectory() as origin_dir:
        logging.info(f'Loading dataset from {repo_id}@{revision}, {dataset_name}, '
                     f'with prefix tags: {prefix_tags!r} ...')
        base_branch_dir = os.path.join(origin_dir, f'branch_{revision}')
        download_archive_as_directory(
            repo_id=repo_id,
            repo_type='dataset',
            revision=revision,
            file_in_repo=f'dataset-{dataset_name}.zip',
            local_directory=base_branch_dir,
        )
        groups = {}
        for group_name in os.listdir(base_branch_dir):
            if os.path.isdir(os.path.join(base_branch_dir, group_name)):
                group_tags = tuple(group_name.split(','))
                if main_group_name:
                    group_tags = tuple([main_group_name, *group_tags])
                tags_to_attach = [*prefix_tags]
                for tag in group_tags:
                    if group_attached_tags.get(tag):
                        tags_to_attach.extend(list(group_attached_tags[tag] or []))
                groups[group_tags] = (
                    os.path.join(base_branch_dir, group_name),
                    tags_to_attach,
                )

        for attached_revision in attach_revisions:
            actual_revision = f'{revision}-{attached_revision}'
            logging.info(f'Loading attached dataset from {repo_id}@{actual_revision}, {dataset_name}, '
                         f'with prefix tags: {prefix_tags!r} ...')
            branch_dir = os.path.join(origin_dir, f'branch_{actual_revision}')
            download_archive_as_directory(
                repo_id=repo_id,
                repo_type='dataset',
                revision=actual_revision,
                file_in_repo=f'dataset-{dataset_name}.zip',
                local_directory=branch_dir,
            )
            for group_name in os.listdir(branch_dir):
                if os.path.isdir(os.path.join(branch_dir, group_name)):
                    group_tags = tuple([attached_revision, *group_name.split(',')])
                    tags_to_attach = [*prefix_tags]
                    for tag in group_tags:
                        if group_attached_tags.get(tag):
                            tags_to_attach.extend(list(group_attached_tags[tag] or []))
                    groups[group_tags] = (
                        os.path.join(branch_dir, group_name),
                        tags_to_attach,
                    )

        with TemporaryDirectory() as dst_dir:
            datasets_arrange(dst_dir, groups, group_weights=group_weights)
            yield dst_dir
