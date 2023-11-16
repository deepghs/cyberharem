import datetime
import glob
import json
import os.path
import zipfile
from typing import Union, Tuple, List, Optional

import pandas as pd
from ditk import logging
from gchar.games import get_character
from gchar.games.base import Character
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from huggingface_hub import CommitOperationAdd, hf_hub_url
from waifuc.action import NoMonochromeAction, FilterSimilarAction, \
    TaggingAction, PersonSplitAction, FaceCountAction, CCIPAction, ModeConvertAction, ClassFilterAction, \
    FileOrderAction, RatingFilterAction, BaseAction, RandomFilenameAction, PaddingAlignAction, ThreeStageSplitAction, \
    AlignMinSizeAction, MinSizeFilterAction, FilterAction
from waifuc.action.filter import MinAreaFilterAction
from waifuc.export import SaveExporter, TextualInversionExporter
from waifuc.model import ImageItem
from waifuc.source import GcharAutoSource, BaseDataSource, LocalSource
from waifuc.utils import task_ctx

from ..utils import number_to_tag, get_ch_name, get_alphabet_name, get_hf_client, download_file, get_hf_fs


def get_source(source) -> BaseDataSource:
    if isinstance(source, (str, Character)):
        source = GcharAutoSource(source, main_sources_count=5)
    elif isinstance(source, BaseDataSource):
        pass
    else:
        raise TypeError(f'Unknown source type - {source!r}.')

    return source


def get_main_source(source, no_r18: bool = False, bg_color: str = 'white',
                    no_monochrome_check: bool = False,
                    drop_multi: bool = True, skip: bool = False) -> BaseDataSource:
    source: BaseDataSource = get_source(source)
    if not skip:
        actions = [ModeConvertAction('RGB', bg_color)]
        if not no_monochrome_check:
            actions.append(NoMonochromeAction())  # no monochrome, greyscale or sketch
        actions.append(ClassFilterAction(['illustration', 'bangumi']))  # no comic or 3d
        if no_r18:
            actions.append(RatingFilterAction(['safe', 'r15']))

        actions.append(FilterSimilarAction('all'))  # filter duplicated images
        if drop_multi:
            actions.append(FaceCountAction(count=1, level='n'))  # drop images with 0 or >1 faces
        actions.extend([
            PersonSplitAction(level='n'),  # crop for each person
            FaceCountAction(count=1, level='n'),
            FileOrderAction(),  # Rename files in order
            CCIPAction(min_val_count=15),  # CCIP, filter the character you may not want to see in dataset
            FilterSimilarAction('all'),  # filter duplicated images
            MinSizeFilterAction(320),
            TaggingAction(force=True, character_threshold=1.01),
        ])
        actions.append(RandomFilenameAction(ext='.png'))
    else:
        actions = []

    return source.attach(*actions)


def actions_parse(actions: Union[int, Tuple[int, int], List[BaseAction]], bg_color: str = 'white'):
    if isinstance(actions, list):
        return actions
    elif isinstance(actions, tuple):
        width, height = actions
        return [PaddingAlignAction((width, height), bg_color)]
    elif isinstance(actions, int):
        return [AlignMinSizeAction(actions)]
    else:
        raise TypeError(f'Unknown post action type - {actions!r}.')


class CustomMinSizeAction(FilterAction):
    def __init__(self, main_size: int = 280, min_eye_size: int = 180):
        self.main_size = main_size
        self.min_eye_size = min_eye_size

    def check(self, item: ImageItem) -> bool:
        min_size = min(item.image.width, item.image.height)
        if 'crop' in item.meta and item.meta['crop']['type'] == 'eye':
            return min_size >= self.min_eye_size
        else:
            return min_size >= self.main_size


_SOURCES = {
    'native': [
        TaggingAction(force=False, character_threshold=1.01),
    ],
    'stage3': [
        ThreeStageSplitAction(split_person=False),
        FilterSimilarAction(),
        MinSizeFilterAction(280),
        TaggingAction(force=False, character_threshold=1.01),
    ],
    'stage3-eyes': [
        ThreeStageSplitAction(split_person=False, split_eyes=True),
        FilterSimilarAction(),
        CustomMinSizeAction(280, 180),
        TaggingAction(force=False, character_threshold=1.01),
    ]
}

_DEFAULT_RESOLUTIONS = {
    'raw': ('native', [], 'Raw data with meta information.'),
    'raw-stage3': ('stage3', [], '3-stage cropped raw data with meta information.'),
    'raw-stage3-eyes': ('stage3-eyes', [], '3-stage cropped (with eye-focus) raw data with meta information.'),
    '384x512': ('native', (384, 512), '384x512 aligned dataset.'),
    # '512x512': ('native', (512, 512), '512x512 aligned dataset.'),
    '512x704': ('native', (512, 704), '512x704 aligned dataset.'),
    # '640x640': ('native', (640, 640), '640x640 aligned dataset.'),
    '640x880': ('native', (640, 880), '640x880 aligned dataset.'),
    'stage3-640': ('stage3', 640, '3-stage cropped dataset with the shorter side not exceeding 640 pixels.'),
    'stage3-800': ('stage3', 800, '3-stage cropped dataset with the shorter side not exceeding 800 pixels.'),
    'stage3-p512-640': ('stage3', [MinAreaFilterAction(512), AlignMinSizeAction(640)],
                        '3-stage cropped dataset with the area not less than 512x512 pixels.'),
    # 'stage3-1200': ('stage3', 1200, '3-stage cropped dataset with the shorter side not exceeding 1200 pixels.'),
    'stage3-eyes-640': ('stage3-eyes', 640, '3-stage cropped (with eye-focus) dataset '
                                            'with the shorter side not exceeding 640 pixels.'),
    'stage3-eyes-800': ('stage3-eyes', 800, '3-stage cropped (with eye-focus) dataset '
                                            'with the shorter side not exceeding 800 pixels.'),
}

DATASET_PVERSION = 'v1.4'


def crawl_dataset_to_huggingface(
        source: Union[str, Character, BaseDataSource], repository: Optional[str] = None,
        name: Optional[str] = None, limit: Optional[int] = 200, min_images: int = 10,
        no_r18: bool = False, bg_color: str = 'white', drop_multi: bool = True, skip_preprocess: bool = False,
        no_monochrome_check: bool = False,
        repo_type: str = 'dataset', revision: str = 'main', path_in_repo: str = '.', private: bool = False,
):
    if isinstance(source, (str, Character)):
        if isinstance(source, str):
            source = get_character(source)
        name = f'{source.enname} ({source.__official_name__})'

        if not repository:
            repository = f'CyberHarem/{get_ch_name(source)}'

    else:
        if name is None:
            raise ValueError('Name must be specified when source is not str or character.')

        if not repository:
            repository = f'CyberHarem/{get_alphabet_name(name)}'

    origin_source = get_main_source(source, no_r18, bg_color, no_monochrome_check, drop_multi, skip_preprocess)
    with TemporaryDirectory() as td:
        # save origin directory
        origin_dir = os.path.join(td, 'origin')
        os.makedirs(origin_dir, exist_ok=True)
        if limit is not None:
            origin_source = origin_source[:limit]
        with task_ctx('origin'):
            origin_source.export(SaveExporter(origin_dir))

        img_count = len(glob.glob(os.path.join(origin_dir, '*.png')))
        if img_count < min_images:
            logging.warn(f'Only {plural_word(img_count, "image")} found for {name} which is too few, '
                         f'skip post-processing and uploading.')
            return

        source_dir = os.path.join(td, 'source')
        os.makedirs(source_dir, exist_ok=True)
        for sname, actions in _SOURCES.items():
            with task_ctx(f'source/{sname}'):
                LocalSource(origin_dir).attach(*actions).export(SaveExporter(os.path.join(source_dir, sname)))

        processed_dir = os.path.join(td, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        archive_dir = os.path.join(td, 'archives')
        os.makedirs(archive_dir, exist_ok=True)

        files_to_upload: List[Tuple[str, str]] = []
        resolutions = _DEFAULT_RESOLUTIONS

        columns = ['Name', 'Images', 'Download', 'Description']
        rows = []
        for rname, (sname, actions, description) in resolutions.items():
            actions = actions_parse(actions, bg_color)

            ox = LocalSource(os.path.join(source_dir, sname))
            current_processed_dir = os.path.join(processed_dir, rname)
            with task_ctx(f'archive/{rname}'):
                if not rname.startswith('raw'):  # raw is preserved for exporting json data
                    ox.attach(*actions).export(TextualInversionExporter(current_processed_dir))
                else:
                    ox.attach(*actions).export(SaveExporter(current_processed_dir))
            current_img_cnt = len(glob.glob(os.path.join(current_processed_dir, '*.png')))

            zip_file = os.path.join(archive_dir, f'dataset-{rname}.zip')
            with zipfile.ZipFile(zip_file, mode='w') as zf:
                for directory, _, files in os.walk(current_processed_dir):
                    for file in files:
                        file_path = os.path.join(directory, file)
                        rel_file_path = os.path.relpath(file_path, current_processed_dir)
                        zf.write(
                            file_path,
                            '/'.join(rel_file_path.split(os.sep))
                        )

            rows.append((
                rname,
                current_img_cnt,
                f'[Download]({os.path.basename(zip_file)})',
                description,
            ))

            files_to_upload.append((zip_file, os.path.basename(zip_file)))

        meta_file = os.path.join(td, 'meta.json')
        with open(meta_file, 'w', encoding='utf-8') as mf:
            json.dump({
                'name': name,
                'version': DATASET_PVERSION,
            }, mf, indent=4, sort_keys=True, ensure_ascii=False)
        files_to_upload.append((meta_file, 'meta.json'))

        readme_file = os.path.join(td, 'README.md')
        with open(readme_file, 'w', encoding='utf-8') as rf:
            print(f'---', file=rf)
            print(f'license: mit', file=rf)
            print(f'task_categories:', file=rf)
            print(f'- text-to-image', file=rf)
            print(f'tags:', file=rf)
            print(f'- art', file=rf)
            print(f'- not-for-all-audiences', file=rf)
            print(f'size_categories:', file=rf)
            print(f'- {number_to_tag(img_count)}', file=rf)
            print(f'---', file=rf)
            print(f'', file=rf)

            print(f'# Dataset of {name}', file=rf)
            print(f'', file=rf)

            print(f'This is the dataset of {name}, '
                  f'containing {plural_word(img_count, "images")} and their tags.', file=rf)
            print(f'', file=rf)

            print(f'Images are crawled from many sites (e.g. danbooru, pixiv, zerochan ...), '
                  f'the auto-crawling system is powered by [DeepGHS Team](https://github.com/deepghs)'
                  f'([huggingface organization](https://huggingface.co/deepghs)).', file=rf)
            print(f'', file=rf)

            df = pd.DataFrame(columns=columns, data=rows)
            print(df.to_markdown(index=False), file=rf)
            print('', file=rf)

        files_to_upload.append((readme_file, 'README.md'))

        hf_client = get_hf_client()
        hf_fs = get_hf_fs()
        logging.info(f'Initialize repository {repository!r}')
        if not hf_fs.exists(f'datasets/{repository}/.gitattributes'):
            hf_client.create_repo(repo_id=repository, repo_type=repo_type, exist_ok=True, private=private)

        current_time = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
        commit_message = f"Publish character {name}, on {current_time}"
        logging.info(f'Publishing character {name!r} to repository {repository!r} ...')
        hf_client.create_commit(
            repository,
            [
                CommitOperationAdd(
                    path_in_repo=f'{path_in_repo}/{filename}',
                    path_or_fileobj=local_file,
                ) for local_file, filename in files_to_upload
            ],
            commit_message=commit_message,
            repo_type=repo_type,
            revision=revision,
            run_as_future=False,
        )


def remake_dataset_to_huggingface(
        repository: Optional[str] = None, limit: Optional[int] = 200, min_images: int = 10,
        no_r18: bool = False, bg_color: str = 'white', drop_multi: bool = True,
        repo_type: str = 'dataset', revision: str = 'main', path_in_repo: str = '.',
):
    hf_fs = get_hf_fs()
    with TemporaryDirectory() as td:
        zip_file = os.path.join(td, 'dataset-raw.zip')
        download_file(hf_hub_url(repository, 'dataset-raw.zip', repo_type='dataset'), zip_file)

        source_dir = os.path.join(td, 'source')
        os.makedirs(source_dir, exist_ok=True)
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(source_dir)

        source = LocalSource(source_dir)
        name = None
        if hf_fs.exists(f'datasets/{repository}/meta.json'):
            meta_json = json.loads(hf_fs.read_text(f'datasets/{repository}/meta.json'))
            if 'name' in meta_json:
                name = meta_json['name']
        name = name or repository.split('/')[-1]
        return crawl_dataset_to_huggingface(
            source, repository, name,
            limit, min_images, no_r18, bg_color, drop_multi, True,
            repo_type, revision, path_in_repo
        )
