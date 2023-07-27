import datetime
import glob
import os.path
import zipfile
from typing import Union, Mapping, Tuple, List, Optional

from ditk import logging
from gchar.games import get_character
from gchar.games.base import Character
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from huggingface_hub import CommitOperationAdd, HfApi
from waifuc.action import NoMonochromeAction, FilterSimilarAction, \
    TaggingAction, PersonSplitAction, FaceCountAction, CCIPAction, ModeConvertAction, ClassFilterAction, \
    FileOrderAction, RatingFilterAction, BaseAction, RandomFilenameAction, PaddingAlignAction
from waifuc.export import SaveExporter, TextualInversionExporter
from waifuc.source import GcharAutoSource, BaseDataSource, LocalSource

from ..utils import number_to_tag


def get_source(source) -> BaseDataSource:
    if isinstance(source, (str, Character)):
        source = GcharAutoSource(source, main_sources_count=5)
    elif isinstance(source, BaseDataSource):
        pass
    else:
        raise TypeError(f'Unknown source type - {source!r}.')

    return source


def get_main_source(source, no_r18: bool = False, bg_color: str = 'white',
                    no_character_tags: bool = True) -> BaseDataSource:
    source: BaseDataSource = get_source(source)
    actions = [
        ModeConvertAction('RGB', bg_color),
        NoMonochromeAction(),  # no monochrome, greyscale or sketch
        ClassFilterAction(['illustration', 'bangumi']),  # no comic or 3d
    ]
    if no_r18:
        actions.append(RatingFilterAction(['safe', 'r15']))

    actions.extend([
        FilterSimilarAction('all'),  # filter duplicated images
        FaceCountAction(count=1),  # drop images with 0 or >1 faces
        PersonSplitAction(),  # crop for each person
        FaceCountAction(count=1),
        FileOrderAction(),  # Rename files in order
        CCIPAction(min_val_count=15),  # CCIP, filter the character you may not want to see in dataset
        FilterSimilarAction('all'),  # filter duplicated images
    ])
    if no_character_tags:
        actions.append(TaggingAction(force=True, character_threshold=1.01))
    else:
        actions.append(TaggingAction(force=True))
    actions.append(RandomFilenameAction(ext='.png'))

    return source.attach(*actions)


def actions_parse(actions: Union[Tuple[int, int], List[BaseAction]], bg_color: str = 'white'):
    if isinstance(actions, list):
        return actions
    elif isinstance(actions, tuple):
        width, height = actions
        return [PaddingAlignAction((width, height), bg_color)]
    else:
        raise TypeError(f'Unknown post action type - {actions!r}.')


_DEFAULT_RESOLUTIONS = {
    'raw': [],
    '384x512': (384, 512),
    '512x512': (512, 512),
    '512x704': (512, 704),
    '640x640': (640, 640),
    '640x880': (640, 880),
}


def crawl_dataset_to_huggingface(
        source: Union[str, Character, BaseDataSource], repository: str, name: Optional[str] = None,
        limit: Optional[int] = 200, min_images: int = 10,
        resolutions: Mapping[str, Union[Tuple[int, int], List[BaseAction]]] = None,
        no_r18: bool = False, bg_color: str = 'white', no_character_tags: bool = True,
        repo_type: str = 'dataset', revision: str = 'main', path_in_repo: str = '.',
):
    if isinstance(source, (str, Character)):
        if isinstance(source, str):
            source = get_character(source)
        name = f'{source.enname} ({source.__official_name__})'
    elif name is None:
        raise ValueError('Name must be specified when source is not str or character.')

    origin_source = get_main_source(source, no_r18, bg_color, no_character_tags)
    with TemporaryDirectory() as td:
        # save origin directory
        origin_dir = os.path.join(td, 'origin')
        os.makedirs(origin_dir, exist_ok=True)
        if limit is not None:
            origin_source = origin_source[:limit]
        origin_source.export(SaveExporter(origin_dir))

        img_count = len(glob.glob(os.path.join(origin_dir, '*.png')))
        if img_count < min_images:
            logging.warn(f'Only {plural_word(img_count, "image")} found for {name} which is too few, '
                         f'skip post-processing and uploading.')
            return

        processed_dir = os.path.join(td, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        archive_dir = os.path.join(td, 'archives')
        os.makedirs(archive_dir, exist_ok=True)

        files_to_upload: List[Tuple[str, str]] = []
        if resolutions is None:
            resolutions = _DEFAULT_RESOLUTIONS
        for rname, actions in resolutions.items():
            actions = actions_parse(actions, bg_color)

            ox = LocalSource(origin_dir)
            current_processed_dir = os.path.join(processed_dir, rname)
            if rname != 'raw':  # raw is preserved for exporting json data
                ox.attach(*actions).export(TextualInversionExporter(current_processed_dir))
            else:
                ox.attach(*actions).export(SaveExporter(current_processed_dir))

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

            files_to_upload.append((zip_file, os.path.basename(zip_file)))

        readme_file = os.path.join(td, 'README.md')
        with open(readme_file, 'w', encoding='utf-8') as rf:
            print(f'---', file=rf)
            print(f'license: mit', file=rf)
            print(f'task_categories:', file=rf)
            print(f'- text-to-image', file=rf)
            print(f'tags:', file=rf)
            print(f'- art', file=rf)
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

        files_to_upload.append((readme_file, 'README.md'))

        hf_client = HfApi(token=os.environ['HF_TOKEN'])
        logging.info(f'Initialize repository {repository!r}')
        hf_client.create_repo(repo_id=repository, repo_type=repo_type, exist_ok=True)

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
        )
