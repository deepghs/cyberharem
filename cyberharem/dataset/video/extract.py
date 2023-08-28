import datetime
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
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from imgutils.data import load_image
from imgutils.metrics import ccip_extract_feature, ccip_batch_differences, ccip_default_threshold
from natsort import natsorted
from sklearn.cluster import OPTICS
from tqdm.auto import tqdm
from waifuc.action import CCIPAction, PaddingAlignAction, PersonSplitAction, FaceCountAction, MinSizeFilterAction, \
    NoMonochromeAction, FilterSimilarAction, HeadCountAction, FileOrderAction
from waifuc.export import SaveExporter
from waifuc.model import ImageItem
from waifuc.source import VideoSource, BaseDataSource, LocalSource
from waifuc.utils import task_ctx

from ...utils import number_to_tag, get_hf_client, get_hf_fs


class ListFeatImageSource(BaseDataSource):
    def __init__(self, image_files, feats):
        self.image_files = image_files
        self.feats = feats

    def _iter(self) -> Iterator[ImageItem]:
        for file, feat in zip(self.image_files, self.feats):
            yield ImageItem(load_image(file), {'ccip_feature': feat, 'filename': os.path.basename(file)})


def cluster_from_directory(src_dir, dst_dir, merge_threshold: float = 0.8, clu_min_samples: int = 5,
                           extract_from_noise: bool = True):
    image_files = np.array(natsorted(glob.glob(os.path.join(src_dir, '*.png'))))

    logging.info(f'Extracting feature of {plural_word(len(image_files), "images")} ...')
    images = [ccip_extract_feature(img) for img in tqdm(image_files, desc='Extract features')]
    l_images = np.stack(images)
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

    # trying to merge clusters
    max_clu_id = labels.max().item()
    logging.info(f'Cluster complete, with {plural_word(max_clu_id, "cluster")}.')
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
                if score >= merge_threshold:
                    labels[labels == yi] = xi
                    logging.info(f'Merging label {yi} into {xi} ...')
                    _exist_ids.remove(yi)
                    _round_merged = True

        if not _round_merged:
            break

    logging.info(f'Merge complete, remained cluster ids: {sorted(_exist_ids)}.')
    ids = []
    for i, clu_id in enumerate(tqdm(sorted(_exist_ids))):
        logging.info(f'Cluster {clu_id} will be renamed as #{i}.')
        logging.info(f'Filtering #{i} from cluster {clu_id} and noise data ...')
        source = ListFeatImageSource(image_files[labels == clu_id], l_images[labels == clu_id])
        if extract_from_noise:
            noise_source = ListFeatImageSource(image_files[labels == -1], l_images[labels == -1])
            source = noise_source.attach(CCIPAction(source, cmp_threshold=0.85))

        with task_ctx(f'#{i}'):
            source.export(SaveExporter(os.path.join(dst_dir, str(i)), no_meta=True))

        if glob.glob(os.path.join(dst_dir, str(i), '*.png')):
            ids.append(i)

    _exist_filenames = set([
        os.path.basename(file) for file in
        glob.glob(os.path.join(dst_dir, '*', '*.png'))
    ])
    noise_dir = os.path.join(dst_dir, '-1')
    os.makedirs(noise_dir, exist_ok=True)
    for file in glob.glob(os.path.join(src_dir, '*.png')):
        if os.path.basename(file) not in _exist_filenames:
            shutil.copyfile(file, os.path.join(noise_dir, os.path.basename(file)))
    noise_cnt = len(glob.glob(os.path.join(noise_dir, '*.png')))
    if noise_cnt > 0:
        logging.info(f'{plural_word(noise_cnt, "noise images")} found.')
        ids.append(-1)
    return ids


def create_project_by_result(bangumi_name: str, ids, clu_dir, dst_dir, preview_count: int = 8):
    total_image_cnt = 0
    columns = ['#', 'Images', 'Download', *(f'Preview {i}' for i in range(1, preview_count + 1))]
    rows = []
    for id_ in ids:
        logging.info(f'Packing for #{id_} ...')
        person_dir = os.path.join(dst_dir, str(id_))
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
                        min_size: int = 320, merge_threshold: float = 0.8, preview_count: int = 8):
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
                           min_size: int = 320, merge_threshold: float = 0.7, preview_count: int = 8):
    logging.info(f'Initializing repository {repository!r} ...')
    hf_client = get_hf_client()
    hf_client.create_repo(repo_id=repository, repo_type='dataset', exist_ok=True)
    hf_fs = get_hf_fs()

    _exist_files = [os.path.relpath(file, repository) for file in hf_fs.glob(f'{repository}/**')]
    _exist_ps = sorted([(file, file.split('/')) for file in _exist_files], key=lambda x: x[1])
    pre_exist_files = set()
    for i, (file, segments) in enumerate(_exist_ps):
        if i < len(_exist_ps) - 1 and segments == _exist_ps[i + 1][1][:len(segments)]:
            continue
        pre_exist_files.add(file)

    with extract_from_videos(video_or_directory, bangumi_name, no_extract,
                             min_size, merge_threshold, preview_count) as dst_dir:
        operations = []
        for directory, _, files in os.walk(dst_dir):
            for file in files:
                filename = os.path.abspath(os.path.join(dst_dir, directory, file))
                file_in_repo = os.path.relpath(filename, dst_dir)
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
        commit_message = f'Publish {bangumi_name}\'s data, on {current_time}'
        logging.info(f'Publishing {bangumi_name}\'s data to repository {repository!r} ...')
        hf_client.create_commit(
            repository,
            operations,
            commit_message=commit_message,
            repo_type='dataset',
            revision=revision,
        )
