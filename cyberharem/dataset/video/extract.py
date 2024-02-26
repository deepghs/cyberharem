import glob
import json
import logging
import os.path
import random
import re
import shutil
import zipfile
from contextlib import contextmanager
from textwrap import dedent
from typing import Iterator

import numpy as np
import pandas as pd
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory
from imgutils.data import load_image
from imgutils.metrics import ccip_extract_feature, ccip_batch_differences, ccip_default_threshold
from natsort import natsorted
from sklearn.cluster import OPTICS
from tqdm.auto import tqdm
from waifuc.action import PaddingAlignAction, PersonSplitAction, FaceCountAction, MinSizeFilterAction, \
    NoMonochromeAction, FilterSimilarAction, HeadCountAction, FileOrderAction, TaggingAction, RandomFilenameAction, \
    BackgroundRemovalAction, ModeConvertAction, FileExtAction
from waifuc.action.filter import MinAreaFilterAction
from waifuc.export import SaveExporter, TextualInversionExporter
from waifuc.model import ImageItem
from waifuc.source import VideoSource, BaseDataSource, LocalSource, EmptySource

from ...utils import number_to_tag, get_hf_client, get_hf_fs


class ListFeatImageSource(BaseDataSource):
    def __init__(self, image_files, feats):
        self.image_files = image_files
        self.feats = feats

    def _iter(self) -> Iterator[ImageItem]:
        for file, feat in zip(self.image_files, self.feats):
            yield ImageItem(load_image(file), {'ccip_feature': feat, 'filename': os.path.basename(file)})


def cluster_from_directory(src_dir, dst_dir, merge_threshold: float = 0.85, clu_min_samples: int = 5,
                           extract_from_noise: bool = True):
    image_files = np.array(natsorted(glob.glob(os.path.join(src_dir, '*.png'))))

    logging.info(f'Extracting feature of {plural_word(len(image_files), "images")} ...')
    images = [ccip_extract_feature(img) for img in tqdm(image_files, desc='Extract features')]
    batch_diff = ccip_batch_differences(images)
    batch_same = batch_diff <= ccip_default_threshold()

    # clustering
    def _metric(x, y):
        return batch_diff[int(x), int(y)].item()

    logging.info('Clustering ...')
    samples = np.arange(len(images)).reshape(-1, 1)
    # max_eps, _ = ccip_default_clustering_params(method='optics_best')
    clustering = OPTICS(min_samples=clu_min_samples, metric=_metric).fit(samples)
    labels = clustering.labels_

    max_clu_id = labels.max().item()
    all_label_ids = np.array([-1, *range(0, max_clu_id + 1)])
    logging.info(f'Cluster complete, with {plural_word(max_clu_id, "cluster")}.')
    label_cnt = {i: (labels == i).sum() for i in all_label_ids if (labels == i).sum() > 0}
    logging.info(f'Current label count: {label_cnt}')

    if extract_from_noise:
        mask_labels = labels.copy()
        for nid in tqdm(np.where(labels == -1)[0], desc='Matching for noises'):
            avg_dists = np.array([
                batch_diff[nid][labels == i].mean()
                for i in range(0, max_clu_id + 1)
            ])
            r_sames = np.array([
                batch_same[nid][labels == i].mean()
                for i in range(0, max_clu_id + 1)
            ])
            best_id = np.argmin(avg_dists)
            if r_sames[best_id] >= 0.90:
                mask_labels[nid] = best_id
        labels = mask_labels
        logging.info('Noise extracting complete.')
        label_cnt = {i: (labels == i).sum() for i in all_label_ids if (labels == i).sum() > 0}
        logging.info(f'Current label count: {label_cnt}')

    # trying to merge clusters
    _exist_ids = set(range(0, max_clu_id + 1))
    while True:
        _round_merged = False
        for xi in range(0, max_clu_id + 1):
            if xi not in _exist_ids:
                continue
            for yi in range(xi + 1, max_clu_id + 1):
                if yi not in _exist_ids:
                    continue

                score = (batch_same[labels == xi][:, labels == yi]).mean()
                logging.info(f'Label {xi} and {yi}\'s similarity score: {score}')
                if score >= merge_threshold:
                    labels[labels == yi] = xi
                    logging.info(f'Merging label {yi} into {xi} ...')
                    _exist_ids.remove(yi)
                    _round_merged = True

        if not _round_merged:
            break

    logging.info(f'Merge complete, remained cluster ids: {sorted(_exist_ids)}.')
    label_cnt = {i: (labels == i).sum() for i in all_label_ids if (labels == i).sum() > 0}
    logging.info(f'Current label count: {label_cnt}')
    ids = []
    for i, clu_id in enumerate(tqdm(sorted(_exist_ids))):
        total = (labels == clu_id).sum()
        logging.info(f'Cluster {clu_id} will be renamed as #{i}, {plural_word(total, "image")} in total.')
        os.makedirs(os.path.join(dst_dir, str(i)), exist_ok=True)
        for imgfile in image_files[labels == clu_id]:
            shutil.copyfile(imgfile, os.path.join(dst_dir, str(i), os.path.basename(imgfile)))
        ids.append(i)

    n_total = (labels == -1).sum()
    if n_total > 0:
        logging.info(f'Save noise images, {plural_word(n_total, "image")} in total.')
        os.makedirs(os.path.join(dst_dir, str(-1)), exist_ok=True)
        for imgfile in image_files[labels == -1]:
            shutil.copyfile(imgfile, os.path.join(dst_dir, str(-1), os.path.basename(imgfile)))
        ids.append(-1)

    return ids


def create_project_by_result(bangumi_name: str, ids, clu_dir, dst_dir, preview_count: int = 8, regsize: int = 2000):
    total_image_cnt = 0
    columns = ['#', 'Images', 'Download', *(f'Preview {i}' for i in range(1, preview_count + 1))]
    rows = []
    reg_source = EmptySource()
    for id_ in ids:
        logging.info(f'Packing for #{id_} ...')
        person_dir = os.path.join(dst_dir, str(id_))
        new_reg_source = LocalSource(os.path.join(clu_dir, str(id_)), shuffle=True).attach(
            MinAreaFilterAction(400)
        )
        reg_source = reg_source | new_reg_source
        os.makedirs(person_dir, exist_ok=True)
        with zipfile.ZipFile(os.path.join(person_dir, 'dataset.zip'), 'w') as zf:
            all_person_images = glob.glob(os.path.join(clu_dir, str(id_), '*.png'))
            total_image_cnt += len(all_person_images)
            for file in all_person_images:
                zf.write(file, os.path.basename(file))

        for i, file in enumerate(random.sample(all_person_images, k=min(len(all_person_images), preview_count)),
                                 start=1):
            PaddingAlignAction((512, 704))(ImageItem(load_image(file))) \
                .image.save(os.path.join(person_dir, f'preview_{i}.png'))

        rel_zip_path = os.path.relpath(os.path.join(person_dir, 'dataset.zip'), dst_dir)
        row = [id_ if id_ != -1 else 'noise', len(all_person_images), f'[Download]({rel_zip_path})']
        for i in range(1, preview_count + 1):
            if os.path.exists(os.path.join(person_dir, f'preview_{i}.png')):
                relpath = os.path.relpath(os.path.join(person_dir, f'preview_{i}.png'), dst_dir)
                row.append(f'![preview {i}]({relpath})')
            else:
                row.append('N/A')
        rows.append(row)

    with TemporaryDirectory() as td:
        logging.info('Creating regular normal dataset ...')
        reg_source.attach(
            TaggingAction(force=False, character_threshold=1.01),
            RandomFilenameAction(),
        )[:regsize].export(TextualInversionExporter(td))

        logging.info('Packing regular normal dataset ...')
        reg_zip = os.path.join(dst_dir, 'regular', 'normal.zip')
        os.makedirs(os.path.dirname(reg_zip), exist_ok=True)
        with zipfile.ZipFile(reg_zip, 'w') as zf:
            for file in glob.glob(os.path.join(td, '*')):
                zf.write(file, os.path.relpath(file, td))

        with TemporaryDirectory() as td_nobg:
            logging.info('Creating regular no-background dataset ...')
            LocalSource(td).attach(
                BackgroundRemovalAction(),
                ModeConvertAction('RGB', 'white'),
                TaggingAction(force=True, character_threshold=1.01),
                FileExtAction('.png'),
            ).export(TextualInversionExporter(td_nobg))

            logging.info('Packing regular no-background dataset ...')
            reg_nobg_zip = os.path.join(dst_dir, 'regular', 'nobg.zip')
            os.makedirs(os.path.dirname(reg_nobg_zip), exist_ok=True)
            with zipfile.ZipFile(reg_nobg_zip, 'w') as zf:
                for file in glob.glob(os.path.join(td_nobg, '*')):
                    zf.write(file, os.path.relpath(file, td_nobg))

    logging.info('Packing all images ...')
    all_zip = os.path.join(dst_dir, 'all.zip')
    with zipfile.ZipFile(all_zip, 'w') as zf:
        for file in glob.glob(os.path.join(clu_dir, '*', '*.png')):
            zf.write(file, os.path.relpath(file, clu_dir))

    logging.info('Packing raw package ...')
    raw_zip = os.path.join(dst_dir, 'raw.zip')
    with zipfile.ZipFile(raw_zip, 'w') as zf:
        for file in glob.glob(os.path.join(clu_dir, '*', '*.png')):
            zf.write(file, os.path.basename(file))

    with open(os.path.join(dst_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'name': bangumi_name,
            'ids': ids,
            'total': total_image_cnt,
        }, f, indent=4, sort_keys=True, ensure_ascii=False)

    with open(os.path.join(dst_dir, 'README.md'), 'w', encoding='utf-8') as f:
        print(dedent(f"""
        ---
        license: mit
        tags:
        - art
        size_categories:
        - {number_to_tag(total_image_cnt)}
        ---
        """).strip(), file=f)
        print('', file=f)

        c_name = ' '.join(map(str.capitalize, re.split(r'\s+', bangumi_name)))
        print(f'# Bangumi Image Base of {c_name}', file=f)
        print('', file=f)

        print(f'This is the image base of bangumi {bangumi_name}, '
              f'we detected {plural_word(len(ids), "character")}, '
              f'{plural_word(total_image_cnt, "images")} in total. '
              f'The full dataset is [here]({os.path.relpath(all_zip, dst_dir)}).', file=f)
        print('', file=f)

        print(f'**Please note that these image bases are not guaranteed to be 100% cleaned, '
              f'they may be noisy actual.** If you intend to manually train models using this dataset, '
              f'we recommend performing necessary preprocessing on the downloaded dataset to eliminate '
              f'potential noisy samples (approximately 1% probability).', file=f)
        print('', file=f)

        print(f'Here is the characters\' preview:', file=f)
        print('', file=f)

        df = pd.DataFrame(columns=columns, data=rows)
        print(df.to_markdown(index=False), file=f)
        print('', file=f)


@contextmanager
def extract_from_videos(video_or_directory: str, bangumi_name: str, no_extract: bool = False,
                        min_size: int = 320, merge_threshold: float = 0.85, preview_count: int = 8):
    if no_extract:
        source = LocalSource(video_or_directory)
    else:
        if os.path.isfile(video_or_directory):
            source = VideoSource(video_or_directory)
        elif os.path.isdir(video_or_directory):
            source = VideoSource.from_directory(video_or_directory)
        else:
            raise TypeError(f'Unknown video - {video_or_directory!r}.')

        source = source.attach(
            NoMonochromeAction(),
            PersonSplitAction(keep_original=False, level='n'),
            FaceCountAction(1, level='n'),
            HeadCountAction(1, level='n'),
            MinSizeFilterAction(min_size),
            FilterSimilarAction('all'),
            FileOrderAction(ext='.png'),
        )

    with TemporaryDirectory() as src_dir:
        logging.info('Extract figures from videos ...')
        source.export(SaveExporter(src_dir, no_meta=True))

        with TemporaryDirectory() as clu_dir:
            logging.info(f'Clustering from {src_dir!r} to {clu_dir!r} ...')
            ids = cluster_from_directory(src_dir, clu_dir, merge_threshold)

            with TemporaryDirectory() as dst_dir:
                create_project_by_result(bangumi_name, ids, clu_dir, dst_dir, preview_count)

                yield dst_dir


def extract_to_huggingface(video_or_directory: str, bangumi_name: str,
                           repository: str, revision: str = 'main', no_extract: bool = False,
                           min_size: int = 320, merge_threshold: float = 0.85, preview_count: int = 8,
                           discord_publish: bool = True):
    logging.info(f'Initializing repository {repository!r} ...')
    hf_client = get_hf_client()
    hf_fs = get_hf_fs()
    if not hf_fs.exists(f'datasets/{repository}/.gitattributes'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', exist_ok=True)

    with extract_from_videos(video_or_directory, bangumi_name, no_extract,
                             min_size, merge_threshold, preview_count) as dst_dir:
        upload_directory_as_directory(
            local_directory=dst_dir,
            repo_id=repository,
            path_in_repo='.',
            repo_type='dataset',
            revision=revision,
            clear=True,
        )

    if discord_publish and 'GH_TOKEN' in os.environ:
        from .discord import send_discord_publish_to_github_action
        send_discord_publish_to_github_action(repository)
