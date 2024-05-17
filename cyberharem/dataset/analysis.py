import logging
from typing import List, Tuple

import numpy as np
from PIL import Image
from hbutils.string import plural_word
from imgutils.tagging import is_basic_character_tag, drop_overlap_tags, get_wd14_tags
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from waifuc.action import ProcessAction
from waifuc.export import SaveExporter
from waifuc.model import ImageItem
from waifuc.source import BaseDataSource, LocalSource
from waifuc.utils import task_ctx


def get_character_core_tags(source: BaseDataSource, threshold: float = 0.35) -> List[str]:
    tags_dict = {}
    total_cnt = 0
    with task_ctx('Extract tags'):
        for item in source:
            total_cnt += 1
            for tag in dict(item.meta.get('tags') or {}).keys():
                tags_dict[tag] = tags_dict.get(tag, 0) + 1

    tags_analysis = {tag: cnt * 1.0 / total_cnt for tag, cnt in sorted(tags_dict.items(), key=lambda x: (-x[1], x[0]))}
    logging.info(f'Analysis of tags: {tags_analysis!r}')
    ch_core_tags = []
    for tag, cnt in sorted(tags_dict.items(), key=lambda x: (-x[1], x[0])):
        ratio = cnt * 1.0 / total_cnt
        if ratio >= threshold and is_basic_character_tag(tag):
            ch_core_tags.append(tag)
    logging.info(f'Selected core tags: {ch_core_tags!r}')

    return ch_core_tags


def get_character_tags_info(directory: str, threshold: float = 0.35, pca_ratio: float = 0.5) \
        -> Tuple[List[str], List[Tuple[List[Image.Image], List[str], np.ndarray]], np.ndarray]:
    tags_dict = {}
    tags_idx = {}
    all_tags = []
    total_cnt = 0
    total_idx = 0
    samples_raw = []
    filenames, embs = [], []
    source = LocalSource(directory)
    with task_ctx('Extract tags'):
        for item in source:
            total_cnt += 1
            sample_tags = []
            filenames.append(item.meta['filename'])
            embs.append(get_wd14_tags(item.image, fmt='embedding'))
            for tag in dict(item.meta.get('tags') or {}).keys():
                tags_dict[tag] = tags_dict.get(tag, 0) + 1
                if tag not in tags_idx:
                    tags_idx[tag] = total_idx
                    all_tags.append(tag)
                    total_idx += 1
                sample_tags.append(tag)

            samples_raw.append((item.image, sample_tags))
    samples_raw = np.array(samples_raw, dtype=object)
    filenames = np.array(filenames)
    embs = np.stack(embs)

    tags_analysis = {tag: cnt * 1.0 / total_cnt for tag, cnt in sorted(tags_dict.items(), key=lambda x: (-x[1], x[0]))}
    logging.info(f'Analysis of tags: {tags_analysis!r}')
    ch_core_tags = []
    for tag, cnt in sorted(tags_dict.items(), key=lambda x: (-x[1], x[0])):
        ratio = cnt * 1.0 / total_cnt
        if ratio >= threshold and is_basic_character_tag(tag):
            ch_core_tags.append(tag)
    ch_core_tags_set = set(ch_core_tags)
    logging.info(f'Selected core tags: {ch_core_tags!r}')

    samples = []
    for _, tags in samples_raw:
        feat = np.zeros((total_idx,), dtype=float)
        for tag in tags:
            if tag not in ch_core_tags_set:
                feat[tags_idx[tag]] = 1.0
        samples.append(feat)
    samples = np.stack(samples)

    embs = embs / np.linalg.norm(embs, axis=-1)[..., None]
    assert len(embs.shape) == 2
    pca = PCA(n_components=min(embs.shape[0], embs.shape[1]))
    pca.fit(embs)

    idx = np.argmin(np.abs(np.cumsum(pca.explained_variance_ratio_) - pca_ratio))
    logging.info(f'PCA components: {idx + 1}')
    pca = PCA(n_components=idx + 1)
    pca_embs = pca.fit_transform(embs)

    cluster = OPTICS(metric='cosine', min_samples=5, xi=0.01)
    cluster.fit(pca_embs)
    mx = np.max(cluster.labels_).item()

    clu_samples = []
    for i in range(0, mx + 1):
        mean_feat = samples[cluster.labels_ == i].mean(axis=0)
        clu_tags: List[str] = [
            tag for tag, value in
            sorted(zip(all_tags, mean_feat.tolist()), key=lambda x: (-x[1], x[0]))
            if value >= threshold
        ]
        embs_i = embs[cluster.labels_ == i, ...]
        clu_tags = drop_overlap_tags(clu_tags)
        clu_images: List[Image.Image] = [img for img, _ in samples_raw[cluster.labels_ == i]]
        clu_samples.append((clu_images, clu_tags, embs_i))
        logging.info(f'Cluster {i}, {plural_word(len(clu_images), "image")}, tags: {clu_tags!r}')

    filename_to_group_id = dict(zip(filenames, cluster.labels_.tolist()))

    class GroupAction(ProcessAction):
        def process(self, item: ImageItem) -> ImageItem:
            return ImageItem(
                item.image,
                meta={
                    **item.meta,
                    'cluster_group': filename_to_group_id[item.meta['filename']]
                },
            )

    with task_ctx('Rewrite Groups'):
        LocalSource(directory).attach(GroupAction()).export(SaveExporter(directory))

    return ch_core_tags, clu_samples, embs
