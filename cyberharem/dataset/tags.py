import json
import os.path
import random
from typing import List

from gchar.games.base import Character

from .load import load_dataset_for_character
from ..utils import load_tags_from_directory, get_ch_name, repr_tags

basic_words = [
    'best quality',
    'masterpiece',
    'highres',
]

generic_neg_words = [
    ('worst quality, low quality', 1.4), ('zombie, sketch, interlocked fingers, comic', 1.1),
    ('full body', 1.1), 'lowres', 'bad anatomy', 'bad hands', 'text', 'error', 'missing fingers', 'extra digit',
    'fewer digits', 'cropped', 'worst quality', 'low quality', 'normal quality', 'jpeg artifacts', 'signature',
    'watermark', 'username', 'blurry', 'white border', ('english text, chinese text', 1.05),
    ('censored, mosaic censoring, bar censor', 1.2),
]


def _free_pos_words(generic_words, name, core_tags):
    return [
        *generic_words,
        (name, 1.15),
        *[key for key, _ in sorted(core_tags.items(), key=lambda x: -x[1])],
    ], generic_neg_words, None, True


def _bikini_pos_words(generic_words, name, core_tags):
    return [
        *generic_words,
        ('night', 1.1),
        ('starry sky', 1.1),
        'beach',
        'beautiful detailed sky',
        ('extremely detailed background', 1.2),
        (name, 1.15),
        ('standing', 1.1),
        'looking at viewer',
        ('bikini', 1.3),
        *[key for key, _ in sorted(core_tags.items(), key=lambda x: -x[1])],
        'light smile',
    ], generic_neg_words, 758691538, True


def _nude_pos_words(generic_words, name, core_tags):
    return [
        'nsfw',
        *generic_words,
        ('lying on bed', 1.1),
        ('extremely detailed background', 1.2),
        ('nude', 1.4),
        ('spread legs', 1.1),
        ('arms up', 1.1),
        'mature',
        (name, 1.15),
        *[key for key, _ in sorted(core_tags.items(), key=lambda x: -x[1])],
        'nipples',
        ('pussy', 1.15),
        ('pussy juice', 1.3),
        'looking at viewer',
        ('embarrassed', 1.1),
        'endured face',
        'feet out of frame',
    ], generic_neg_words, 465191133, False


def _nude_stand_words(generic_words, name, core_tags):
    return [
        'nsfw',
        *generic_words,
        ('simple background', 1.1),
        ('standing', 1.15),
        ('nude', 1.4),
        ('completely nude', 1.2),
        'mature',
        (name, 1.15),
        *[key for key, _ in sorted(core_tags.items(), key=lambda x: -x[1])],
        'nipples',
        ('pussy', 1.15),
        ('pussy juice', 1.3),
        'looking at viewer',
        ('embarrassed', 1.1),
    ], generic_neg_words, 758691538, False


def _safe_maid_words(generic_words, name, core_tags):
    return [
        *generic_words,
        ('maid', 1.4),
        ('long maid dress', 1.15),
        (name, 1.15),
        *[key for key, _ in sorted(core_tags.items(), key=lambda x: -x[1])],
    ], [
        'nsfw', 'sexy', 'underwear', 'bra', 'fishnet',
        'skin of legs', 'bare legs', 'bare skin', 'navel',
        *generic_neg_words,
    ], None, True


def _safe_yukata_words(generic_words, name, core_tags):
    return [
        *generic_words,
        ('yukata', 1.4),
        ('kimono', 1.2),
        (name, 1.15),
        *[key for key, _ in sorted(core_tags.items(), key=lambda x: -x[1])],
    ], [
        'nsfw', 'sexy', 'underwear', 'bra', 'fishnet',
        'skin of legs', 'bare legs', 'bare skin', 'navel',
        *generic_neg_words,
    ], None, True


def _safe_miko_words(generic_words, name, core_tags):
    return [
        *generic_words,
        ('white kimono', 1.35),
        ('red hakama', 1.35),
        ('wide sleeves', 1.2),
        (name, 1.15),
        *[key for key, _ in sorted(core_tags.items(), key=lambda x: -x[1])],
    ], [
        'nsfw', 'sexy', 'underwear', 'bra', 'fishnet',
        'skin of legs', 'bare legs', 'bare skin', 'navel',
        *generic_neg_words,
    ], None, True


def _safe_suit_words(generic_words, name, core_tags):
    return [
        *generic_words,
        ('black business suit', 1.4),
        ('tie', 1.2),
        ('sunglasses', 1.25),
        ('white gloves', 1.15),
        ('white shirt', 1.1),
        ('black skirt', 1.15),
        ('smoking', 1.2),
        'handsome',
        (name, 1.15),
        *[key for key, _ in sorted(core_tags.items(), key=lambda x: -x[1])],
    ], [
        'nsfw', 'sexy', 'underwear', 'bra', 'fishnet',
        'skin of legs', 'bare legs', 'bare skin', 'navel',
        *generic_neg_words,
    ], None, True


EXTRAS = [
    ('free', _free_pos_words),
    ('bikini', _bikini_pos_words),
    ('maid', _safe_maid_words),
    ('miko', _safe_miko_words),
    ('yukata', _safe_yukata_words),
    ('nude', _nude_pos_words),
    ('nude2', _nude_stand_words),
    ('suit', _safe_suit_words),
]


def save_recommended_tags(source, name: str = None, workdir: str = None, ds_size: str = '512x704'):
    with load_dataset_for_character(source, ds_size) as (ch, ds_dir):
        if ch is None:
            if name is None:
                raise ValueError(f'Name should be specified when using custom source - {source!r}.')
        else:
            name = name or get_ch_name(ch)

        workdir = workdir or os.path.join('runs', name)
        tags_dir = os.path.join(workdir, 'rtags')
        os.makedirs(tags_dir, exist_ok=True)

        generic_words = []
        generic_words.extend(basic_words)
        if isinstance(ch, Character):
            if ch.gender == 'male':
                generic_words.extend(['1boy', 'solo'])
            elif ch.gender == 'female':
                generic_words.extend(['1girl', 'solo'])
            else:
                generic_words.append('solo')
        else:
            generic_words.append('solo')

        core_tags, feats = load_tags_from_directory(ds_dir)
        for i, f in enumerate(feats, start=1):
            pos_words = [*generic_words, (name, 1.15), *f.keys()]
            pos_prompt = repr_tags(pos_words)
            neg_prompt = repr_tags(generic_neg_words)

            tags_name = f'pattern_{i}'
            with open(os.path.join(tags_dir, f'{tags_name}.json'), 'w', encoding='utf-8') as f:
                json.dump({
                    'name': tags_name,
                    'prompt': pos_prompt,
                    'neg_prompt': neg_prompt,
                    'seed': random.randint(0, 1 << 31),
                    'sfw': True,
                }, f, indent=4, ensure_ascii=False)

        for tags_name, _func in EXTRAS:
            pos_words, neg_words, seed, is_sfw = _func(generic_words, name, core_tags)
            pos_prompt = repr_tags(pos_words)
            neg_prompt = repr_tags(neg_words)

            with open(os.path.join(tags_dir, f'{tags_name}.json'), 'w', encoding='utf-8') as f:
                json.dump({
                    'name': tags_name,
                    'prompt': pos_prompt,
                    'neg_prompt': neg_prompt,
                    'seed': seed if seed is not None else random.randint(0, 1 << 31),
                    'sfw': is_sfw,
                }, f, indent=4, ensure_ascii=False)


def sort_draw_names(names: List[str]) -> List[str]:
    vs = []
    for name in names:
        if name.startswith('pattern_'):
            vs.append((0, int(name.split('_')[1]), name))
        else:
            vs.append((1, name, name))

    return [item[2] for item in sorted(vs)]
