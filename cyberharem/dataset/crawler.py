import glob
import json
import os.path
import random
from typing import Union, Tuple, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from ditk import logging
from gchar.games import get_character
from gchar.games.base import Character
from hbutils.scale import size_to_bytes_str
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.archive import archive_pack
from hfutils.operate import upload_directory_as_directory, download_archive_as_directory
from huggingface_hub import hf_hub_url
from imgutils.metrics import anime_dbaesthetic
from imgutils.restore import restore_with_scunet
from imgutils.validate import anime_portrait
from waifuc.action import NoMonochromeAction, FilterSimilarAction, \
    TaggingAction, PersonSplitAction, FaceCountAction, CCIPAction, ModeConvertAction, ClassFilterAction, \
    FileOrderAction, RatingFilterAction, BaseAction, RandomFilenameAction, PaddingAlignAction, ThreeStageSplitAction, \
    AlignMinSizeAction, MinSizeFilterAction, FilterAction, MinAreaFilterAction, SafetyAction, TagDropAction, \
    TagOverlapDropAction, AlignMaxAreaAction, BlacklistedTagDropAction, TagRemoveUnderlineAction, ProcessAction
from waifuc.export import SaveExporter, TextualInversionExporter
from waifuc.model import ImageItem
from waifuc.source import GcharAutoSource, BaseDataSource, LocalSource
from waifuc.utils import task_ctx

from .analysis import get_character_tags_info
from .discord import send_discord_publish_to_github_action
from ..utils import number_to_tag, get_ch_name, get_alphabet_name, get_hf_client, get_hf_fs, get_formal_title, \
    get_global_namespace


def get_source(source, drop_multi: bool = False) -> BaseDataSource:
    if isinstance(source, (str, Character)):
        source = GcharAutoSource(
            source,
            main_sources_count=5,
            strict_for_main=drop_multi,
            max_preset_limit=None,
            preset_sites=(
                'zerochan',
                'anime_pictures',
                # 'danbooru',
            ),
            blacklist_sites=('lolibooru',),
            min_size=1200,
        )
    elif isinstance(source, BaseDataSource):
        pass
    else:
        raise TypeError(f'Unknown source type - {source!r}.')

    return source


class RestoreAction(ProcessAction):
    def __init__(self, min_size: int, restore: bool = True):
        self.min_size = min_size
        self.restore = restore

    def process(self, item: ImageItem) -> ImageItem:
        image = item.image
        if self.restore:
            image = restore_with_scunet(image)
        if image.width * image.height <= self.min_size ** 2:
            r = (self.min_size ** 2 / (image.width * image.height)) ** 0.5
            image = image.resize((int(image.width * r), int(image.height * r)), resample=Image.LANCZOS)

        return ImageItem(image, item.meta)


class EmptyAction(ProcessAction):
    def process(self, item: ImageItem) -> ImageItem:
        return item


def get_main_source(source, no_r18: bool = False, bg_color: str = 'white',
                    no_monochrome_check: bool = False, drop_multi: bool = False, skip: bool = False,
                    tiny_scale: float = 1.0, min_resolution: Optional[int] = None) -> BaseDataSource:
    source: BaseDataSource = get_source(source, drop_multi)
    if not skip:
        actions = [
            AlignMaxAreaAction(4500),  # IMPORTANT!!! crawler will crash because of large image if remove this
            ModeConvertAction('RGB', bg_color)
        ]
        if not no_monochrome_check:
            actions.append(NoMonochromeAction())  # no monochrome, greyscale or sketch
        actions.append(SafetyAction())
        actions.append(ClassFilterAction(['illustration', 'bangumi']))  # no comic or 3d
        if no_r18:
            actions.append(RatingFilterAction(['safe', 'r15']))

        actions.append(FilterSimilarAction('all'))  # filter duplicated images
        if drop_multi:
            actions.append(FaceCountAction(1, level='n'))  # drop images with 0 or >1 faces
        actions.extend([
            PersonSplitAction(level='n'),  # crop for each person
            AlignMinSizeAction(1400),
            FaceCountAction(1, level='n'),
            FileOrderAction(),  # Rename files in order
            CCIPAction(min_val_count=15),  # CCIP, filter the character you may not want to see in dataset
            FilterSimilarAction('all', capacity=2000),  # filter duplicated images
            MinSizeFilterAction(int(180 * (tiny_scale ** 0.5))),
            MinAreaFilterAction(int(360 * tiny_scale)),
            RestoreAction(min_resolution) if min_resolution is not None else EmptyAction(),
            TaggingAction(force=True, character_threshold=1.01),
            RandomFilenameAction(ext='.png')
        ])
    else:
        actions = [
            TaggingAction(force=False, character_threshold=1.01),
        ]

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


class UnescapeTagAction(ProcessAction):
    def process(self, item: ImageItem) -> ImageItem:
        tags = dict(item.meta.get('tags') or {})
        tags = {tag.replace('\\', ''): score for tag, score in tags.items()}
        return ImageItem(item.image, {**item.meta, 'tags': tags})


class DirectorySepAction(ProcessAction):
    def process(self, item: ImageItem) -> ImageItem:
        type_, _ = anime_portrait(item.image)
        if type_ == 'head':
            subdir = 'head'
        else:
            subdir = 'others'
        item.meta['filename'] = os.path.join(subdir, item.meta['filename'])
        return item


class AestheticAction(ProcessAction):
    def process(self, item: ImageItem) -> ImageItem:
        level_, _ = anime_dbaesthetic(item.image)
        if level_ == 'masterpiece':
            tag = level_
        else:
            tag = f'{level_} quality'

        item.meta['aesthetic'] = tag
        if 'tags' not in item.meta:
            item.meta['tags'] = {}
        item.meta['tags'][tag] = 1000.0
        return item


DATASET_PVERSION = 'v1.6-alpha0'


def crawl_dataset_to_huggingface(
        source: Union[str, Character, BaseDataSource], repository: Optional[str] = None,
        name: Optional[str] = None, display_name: Optional[str] = None,
        limit: Optional[int] = 500, min_images: int = 10,
        no_r18: bool = False, bg_color: str = 'white', drop_multi: bool = False, skip_preprocess: bool = False,
        no_monochrome_check: bool = False, repo_type: str = 'dataset', revision: str = 'main',
        path_in_repo: str = '.', private: bool = False, n_img_samples: int = 5,
        bangumi_source_repository: Optional[str] = None, remove_empty_repo: bool = True,
        tiny_scale: Optional[float] = 1.0, min_resolution: Optional[int] = None, discord_publish: bool = True,
):
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()
    if isinstance(source, (str, Character)):
        if isinstance(source, str):
            source, origin_source = get_character(source), source
            if not source:
                raise ValueError(f'Character {origin_source!r} not found.')

        name = name or get_ch_name(source)
        display_name = display_name or get_formal_title(source)

        if not repository:
            repository = f'{get_global_namespace()}/{get_ch_name(source)}'

    else:
        if name is None:
            raise ValueError('Name must be specified when source is not str or character.')

        display_name = display_name or name
        if not repository:
            repository = f'{get_global_namespace()}/{get_alphabet_name(name)}'

    try:
        logging.info(f'Repository: {repository!r}, name: {name!r}, display_name: {display_name!r}')
        if not hf_client.repo_exists(repo_id=repository, repo_type=repo_type):
            hf_client.create_repo(repo_id=repository, repo_type=repo_type, exist_ok=True, private=private)
            repo_new_created = True
        else:
            repo_new_created = False
        if not hf_client.revision_exists(repo_id=repository, repo_type=repo_type, revision=revision):
            hf_client.create_branch(repo_id=repository, repo_type=repo_type, branch=revision, exist_ok=True)
        hf_fs.write_text(f'datasets/{repository}/.git-ongoing', 'on-going mark')

        origin_source = get_main_source(
            source=source,
            no_r18=no_r18,
            bg_color=bg_color,
            no_monochrome_check=no_monochrome_check,
            drop_multi=drop_multi,
            skip=skip_preprocess,
            min_resolution=min_resolution,
        )
        with TemporaryDirectory() as td, TemporaryDirectory() as upload_td:
            # save origin directory
            origin_dir = os.path.join(td, 'origin')
            os.makedirs(origin_dir, exist_ok=True)
            if limit is not None:
                origin_source = origin_source[:limit]
            with task_ctx('origin'):
                origin_source.export(SaveExporter(origin_dir))

            # count for images
            img_count = len(glob.glob(os.path.join(origin_dir, '*.png')))
            if img_count < min_images:
                logging.warn(f'Only {plural_word(img_count, "image")} found for {name} which is too few, '
                             f'skip post-processing and uploading.')
                if remove_empty_repo and repo_new_created and \
                        hf_client.repo_exists(repo_id=repository, repo_type=repo_type):
                    hf_client.delete_repo(repo_id=repository, repo_type=repo_type)
                else:
                    hf_fs.write_text(f'datasets/{repository}/.git-empty', 'empty')
                    hf_fs.delete(f'datasets/{repository}/.git-ongoing')
                return

            ch_core_tags, clu_samples = get_character_tags_info(LocalSource(origin_dir))
            all_tags, all_tags_set = [], set()
            for i, (images, tags) in enumerate(clu_samples):
                for tag in tags:
                    if tag not in all_tags_set:
                        all_tags_set.add(tag)
                        all_tags.append(tag)
            samples_dir = os.path.join(upload_td, 'samples')
            os.makedirs(samples_dir, exist_ok=True)
            tgs_columns = ['#', 'Samples', *(f'Img-{i}' for i in range(1, n_img_samples + 1)), 'Tags']
            tgs_rows = []
            tg_tb_columns = ['#', 'Samples', *(f'Img-{i}' for i in range(1, n_img_samples + 1)), *all_tags]
            tg_tb_rows = []
            info_clus = []
            for i, (images, tags) in enumerate(clu_samples):
                clu_size = len(images)
                if len(images) > n_img_samples:
                    images = random.sample(images, k=n_img_samples)
                img_files = []
                for j, image in enumerate(images):
                    dst_file = os.path.join(samples_dir, f'{i}', f'clu{i}-sample{j}.png')
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    image = PaddingAlignAction((512, 704)).process(ImageItem(image)).image
                    image.save(dst_file)
                    img_files.append(hf_hub_url(
                        repo_id=repository,
                        repo_type=repo_type,
                        filename=os.path.relpath(dst_file, upload_td),
                        revision=revision,
                    ))
                tgs_rows.append([
                    i, clu_size,
                    *[f'![]({img_files[i_]})' for i_ in range(n_img_samples)],
                    ', '.join(tags)
                ])
                tg_tb_rows.append([
                    i, clu_size,
                    *[f'![]({img_files[i_]})' for i_ in range(n_img_samples)],
                    *['X' if tag in tags else '' for tag in all_tags],
                ])
                info_clus.append({
                    'id': i,
                    'size': clu_size,
                    'tags': tags,
                })

            source_dir = os.path.join(td, 'source')
            os.makedirs(source_dir, exist_ok=True)
            _SOURCES = {
                'raw': (
                    [
                        TaggingAction(force=False, character_threshold=1.01),
                        AestheticAction(),
                    ], False
                ),
                'stage3': (
                    [
                        ThreeStageSplitAction(split_person=False),
                        FilterSimilarAction(),
                        RestoreAction(min_resolution,
                                      restore=False) if min_resolution is not None else EmptyAction(),
                        MinSizeFilterAction(int(180 * (tiny_scale ** 0.5))),
                        MinAreaFilterAction(int(360 * tiny_scale)),
                        AlignMaxAreaAction(1800),
                        TaggingAction(force=False, character_threshold=1.01),
                        AestheticAction(),
                    ], True
                ),
            }
            for sname, (actions, need_prune) in _SOURCES.items():
                with task_ctx(f'source/{sname}'):
                    if need_prune:
                        actions = [
                            *actions,
                            TagOverlapDropAction(),
                            TagDropAction(ch_core_tags),
                            UnescapeTagAction(),
                            BlacklistedTagDropAction(),
                            TagRemoveUnderlineAction(),
                        ]
                    LocalSource(origin_dir).attach(*actions).export(SaveExporter(os.path.join(source_dir, sname)))

            processed_dir = os.path.join(td, 'processed')
            os.makedirs(processed_dir, exist_ok=True)
            archive_dir = os.path.join(td, 'archives')
            os.makedirs(archive_dir, exist_ok=True)

            _DEFAULT_NORMAL_RESOLUTIONS = {
                'raw': ('raw', True, [], 'Raw data with meta information (min edge aligned to 1400 if larger).'),
                'stage3-p480-1200': ('stage3', False, [
                    MinAreaFilterAction(480),
                    AlignMinSizeAction(1200),
                    DirectorySepAction(),
                ], '3-stage cropped dataset with the area not less than 480x480 pixels.'),
            }
            _DEFAULT_TINY_RESOLUTIONS = {
                'raw': ('raw', True, [], 'Raw data with meta information (min edge aligned to 1400 if larger).'),
                'stage3-p180-1200': ('stage3', False, [
                    MinAreaFilterAction(180),
                    AlignMinSizeAction(1200),
                    DirectorySepAction(),
                ], '3-stage cropped dataset with the area not less than 180x180 pixels.'),
            }
            resolutions = _DEFAULT_NORMAL_RESOLUTIONS if tiny_scale >= 0.75 else _DEFAULT_TINY_RESOLUTIONS

            ds_rows = []
            info_packages = {}
            for rname, (sname, is_raw, actions, description) in resolutions.items():
                actions = actions_parse(actions, bg_color)

                ox = LocalSource(os.path.join(source_dir, sname))
                current_processed_dir = os.path.join(processed_dir, rname)
                with task_ctx(f'archive/{rname}'):
                    if not is_raw:  # raw is preserved for exporting json data
                        ox.attach(*actions).export(TextualInversionExporter(current_processed_dir))
                    else:
                        ox.attach(*actions).export(SaveExporter(current_processed_dir))
                current_img_cnt = len(glob.glob(os.path.join(current_processed_dir, '**', '*.png'), recursive=True))
                current_img_cnt_maps = {}
                for d in os.listdir(current_processed_dir):
                    if os.path.isdir(os.path.join(current_processed_dir, d)):
                        current_img_cnt_maps[d] = len(glob.glob(
                            os.path.join(current_processed_dir, d, '**', '*.png'), recursive=True))
                zip_file = os.path.join(upload_td, f'dataset-{rname}.zip')
                archive_pack('zip', directory=current_processed_dir, archive_file=zip_file, clear=True)
                info_packages[rname] = {
                    'filename': os.path.relpath(zip_file, upload_td),
                    'size': current_img_cnt,
                    'sub_sizes': current_img_cnt_maps,
                    'package_size': os.path.getsize(zip_file),
                    'type': 'Waifuc-Raw' if is_raw else 'IMG+TXT',
                    'description': description,
                }
                zip_download_url = hf_hub_url(
                    repo_id=repository,
                    repo_type=repo_type,
                    filename=os.path.relpath(zip_file, upload_td),
                    revision=revision,
                )

                ds_rows.append({
                    'Name': rname,
                    'Images': current_img_cnt,
                    **{f'Images-{d}': str(cnt) for d, cnt in current_img_cnt_maps.items()},
                    'Size': size_to_bytes_str(os.path.getsize(zip_file), precision=2),
                    'Download': f'[Download]({zip_download_url})',
                    'Type': 'Waifuc-Raw' if is_raw else 'IMG+TXT',
                    'Description': description,
                })

            with open(os.path.join(upload_td, 'meta.json'), 'w', encoding='utf-8') as mf:
                json.dump({
                    'name': name,
                    'display_name': display_name,
                    'bangumi': bangumi_source_repository,
                    'version': DATASET_PVERSION,
                    'base_size': img_count,
                    'packages': info_packages,
                    'core_tags': ch_core_tags,
                    'clusters': info_clus,
                }, mf, indent=4, sort_keys=True, ensure_ascii=False)

            with open(os.path.join(upload_td, 'README.md'), 'w', encoding='utf-8') as rf:
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

                print(f'# Dataset of {display_name}', file=rf)
                print(f'', file=rf)

                print(f'This is the dataset of {display_name}, '
                      f'containing {plural_word(img_count, "images")} and their tags.', file=rf)
                print(f'', file=rf)

                print(f'The core tags of this character are `{", ".join(ch_core_tags)}`, '
                      f'which are pruned in this dataset.', file=rf)
                print(f'', file=rf)

                print(f'Images are crawled from many sites (e.g. danbooru, pixiv, zerochan ...), '
                      f'the auto-crawling system is powered by [DeepGHS Team](https://github.com/deepghs)'
                      f'([huggingface organization](https://huggingface.co/deepghs)).', file=rf)
                print(f'', file=rf)

                print('## List of Packages', file=rf)
                print(f'', file=rf)
                ds_df = pd.DataFrame(data=ds_rows)
                ds_df = ds_df.replace(np.NaN, '--')
                print(ds_df.to_markdown(index=False), file=rf)
                print('', file=rf)

                print('### Load Raw Dataset with Waifuc', file=rf)
                print(f'', file=rf)
                print(f'We provide raw dataset (including tagged images) for '
                      f'[waifuc](https://deepghs.github.io/waifuc/main/tutorials/installation/index.html) '
                      f'loading. If you need this, just run the following code', file=rf)
                print(f'', file=rf)
                print(f'```python', file=rf)
                print(f'import os', file=rf)
                print(f'import zipfile', file=rf)
                print(f'', file=rf)
                print(f'from huggingface_hub import hf_hub_download', file=rf)
                print(f'from waifuc.source import LocalSource', file=rf)
                print(f'', file=rf)
                print(f'# download raw archive file', file=rf)
                print(f'zip_file = hf_hub_download(', file=rf)
                print(f"    repo_id={repository!r},", file=rf)
                print(f"    repo_type={repo_type!r},", file=rf)
                print(f"    filename='dataset-raw.zip',", file=rf)
                print(f')', file=rf)
                print(f'', file=rf)
                print(f'# extract files to your directory', file=rf)
                print(f"dataset_dir = 'dataset_dir'", file=rf)
                print(f'os.makedirs(dataset_dir, exist_ok=True)', file=rf)
                print(f"with zipfile.ZipFile(zip_file, 'r') as zf:", file=rf)
                print(f'    zf.extractall(dataset_dir)', file=rf)
                print(f'', file=rf)
                print(f'# load the dataset with waifuc', file=rf)
                print(f'source = LocalSource(dataset_dir)', file=rf)
                print(f'for item in source:', file=rf)
                print(f"    print(item.image, item.meta['filename'], item.meta['tags'])", file=rf)
                print(f'```', file=rf)
                print(f'', file=rf)

                print('## List of Clusters', file=rf)
                print(f'', file=rf)
                print(f'List of tag clustering result, maybe some outfits can be mined here.', file=rf)
                print(f'', file=rf)
                print('### Raw Text Version', file=rf)
                print(f'', file=rf)
                tgs_df = pd.DataFrame(columns=tgs_columns, data=tgs_rows)
                print(tgs_df.to_markdown(index=False), file=rf)
                print('', file=rf)
                print('### Table Version', file=rf)
                print(f'', file=rf)
                tg_tb_df = pd.DataFrame(columns=tg_tb_columns, data=tg_tb_rows)
                print(tg_tb_df.to_markdown(index=False), file=rf)
                print('', file=rf)

            hf_client = get_hf_client()
            if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
                hf_client.create_repo(repo_id=repository, repo_type='dataset', exist_ok=True, private=private)

            # noinspection PyTypeChecker
            upload_directory_as_directory(
                repo_id=repository,
                repo_type=repo_type,
                local_directory=upload_td,
                path_in_repo=path_in_repo,
                message=f'Publish character {name!r} to repository',
                revision=revision,
                clear=True,
            )

            if discord_publish and 'GH_TOKEN' in os.environ:
                send_discord_publish_to_github_action(repository)

    finally:
        if hf_fs.exists(f'datasets/{repository}/.git-ongoing'):
            hf_fs.rm(f'datasets/{repository}/.git-ongoing')


def remake_dataset_to_huggingface(
        repository: Optional[str] = None, min_images: int = 10,
        no_r18: bool = False, bg_color: str = 'white', drop_multi: bool = True,
        repo_type: str = 'dataset', revision: str = 'main', path_in_repo: str = '.',
        private: bool = False, n_img_samples: int = 5,
):
    hf_fs = get_hf_fs()
    with TemporaryDirectory() as td:
        source_dir = os.path.join(td, 'source')
        os.makedirs(source_dir, exist_ok=True)
        download_archive_as_directory(
            repo_id=repository,
            repo_type=repo_type,
            file_in_repo='dataset-raw.zip',
            local_directory=source_dir,
        )

        source = LocalSource(source_dir)
        name = None
        if hf_fs.exists(f'datasets/{repository}/meta.json'):
            meta_json = json.loads(hf_fs.read_text(f'datasets/{repository}/meta.json'))
            if 'name' in meta_json:
                name = meta_json['name']
            display_name = meta_json.get('display_name')
            bangumi_source_repo = meta_json.get('bangumi')

        name = name or repository.split('/')[-1]
        return crawl_dataset_to_huggingface(
            source=source,
            repository=repository,
            name=name,
            display_name=display_name or name,
            limit=None,
            min_images=min_images,
            no_r18=no_r18,
            bg_color=bg_color,
            drop_multi=drop_multi,
            skip_preprocess=True,
            no_monochrome_check=False,
            repo_type=repo_type,
            revision=revision,
            path_in_repo=path_in_repo,
            private=private,
            n_img_samples=n_img_samples,
            bangumi_source_repository=bangumi_source_repo,
        )
