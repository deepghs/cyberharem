import glob
import os
import pathlib
import re
from typing import Mapping, List, Tuple, Union

import numpy as np
from sklearn.cluster import OPTICS

_CORE_WORDS = [
    'skin', 'eye', 'eyes', 'pupil', 'pupils', 'hair', 'horn', 'horns', 'ear', 'ears', 'neck',
    'breast', 'breasts', 'scar', 'scars', 'face', 'faces', 'blood', 'bleed', 'teeth', 'tooth',
]
_BLACKLISTED_WORDS = [
    'solo', '1girl', '1boy', 'background', 'quality', 'chibi', 'monochrome', 'comic',
    'dress', 'dresses', 'minidress', 'skirt', 'skirts', 'shoulder', 'shoulders',
    'slit', 'gown', 'sundress', 'sweater', 'wedding', 'socks', 'kneehighs',
    'thighhighs', 'pantyhose', 'legwear', 'trousers', 'shorts',
    'bra', 'pantsu', 'panty', 'panties', 'weapon', 'weapons', 'armor',
    'penis', 'pussy', 'vagina', 'clitoris', 'nipple', 'nipples',
    'looking', 'jacket', 'sleeves', 'clothes', 'shirt',
    'glove', 'gloves', 'mask', 'masks', 'coat', 'coats', 'frill', 'frills',
    'costume', 'costumes', 'pant', 'pants',
]


def _contains_core_word(tag: str):
    words = [word for word in re.split(r'[\W_]+', tag.lower()) if word]
    return any(word in _CORE_WORDS for word in words)


def _contains_blacklisted_word(tag: str):
    words = [word for word in re.split(r'[\W_]+', tag.lower()) if word]
    return any(word in _BLACKLISTED_WORDS for word in words)


def find_core_tags(tags: Mapping[str, float], core_threshold: float = 0.25, threshold: float = 0.45) \
        -> Mapping[str, float]:
    retval = {}
    for tag, score in sorted(tags.items(), key=lambda x: (-x[1], x[0])):
        if _contains_blacklisted_word(tag):
            continue

        if score >= threshold or (_contains_core_word(tag) and score >= core_threshold):
            retval[tag] = score

    return retval


def load_tags_from_directory(directory: str, core_threshold: float = 0.25, threshold: float = 0.45) \
        -> Tuple[Mapping[str, float], List[Mapping[str, float]]]:
    all_words = set()
    ids_, word_lists = [], []
    for txt_file in glob.glob(os.path.join(directory, '*.txt')):
        id_ = os.path.splitext(os.path.basename(txt_file))[0]
        origin_text = pathlib.Path(txt_file).read_text().strip()
        words = [word.strip() for word in re.split(r'\s*,\s*', origin_text) if word.strip()]
        ids_.append(id_)
        word_lists.append(words)

        for word in words:
            all_words.add(word)

    all_words = sorted(all_words)
    all_words_map = {word: i for i, word in enumerate(all_words)}

    features = []
    for words in word_lists:
        feat = np.zeros((len(all_words),), dtype=float)
        for word in words:
            feat[all_words_map[word]] = 1.0
        features.append(feat)

    features = np.stack(features)
    mf = features.mean(axis=0)
    all_wds = {
        word: value for word, value in
        sorted(zip(all_words, mf.tolist()), key=lambda x: (-x[1], x[0]))
    }
    core_tags = find_core_tags(all_wds, core_threshold, threshold)

    cluster = OPTICS(metric='cosine', min_samples=5, xi=0.01)
    cluster.fit(features)
    mx = np.max(cluster.labels_).item()

    feats = []
    for i in range(1, mx + 1):
        mean_feat = features[cluster.labels_ == i].mean(axis=0)
        wds = {
            word: value for word, value in
            sorted(zip(all_words, mean_feat.tolist()), key=lambda x: (-x[1], x[0]))
            if value >= threshold
        }
        feats.append({**{key: 1.0 for key in core_tags.keys()}, **wds})

    return core_tags, feats


def repr_tags(tags: List[Union[str, Tuple[str, float]]]) -> str:
    _exists = set()
    _str_items = []
    for item in tags:
        if isinstance(item, tuple):
            tag, weight = item
        else:
            tag, weight = item, None
        if tag in _exists:
            continue

        if weight is not None:
            _str_items.append(f'{{{tag}:{weight:.2f}}}')
        else:
            _str_items.append(tag)
        _exists.add(tag)

    return ', '.join(_str_items)
