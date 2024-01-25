import json
import math
import os
import random
from typing import Iterator, Tuple

from cyberharem.utils import repr_tags

generic_words = [
    'best quality',
    'masterpiece',
    'highres',
    'solo',
]

generic_neg_words = [
    ('worst quality, low quality', 1.4), ('zombie, sketch, interlocked fingers, comic', 1.1),
    ('full body', 1.1), 'lowres', 'bad anatomy', 'bad hands', 'text', 'error', 'missing fingers', 'extra digit',
    'fewer digits', 'cropped', 'worst quality', 'low quality', 'normal quality', 'jpeg artifacts', 'signature',
    'watermark', 'username', 'blurry', 'white border', ('english text, chinese text', 1.05),
    ('censored, mosaic censoring, bar censor', 1.2),
]

_DEFAULT_NAME_WEIGHT = 0.9


def _full_body_words(name):
    return [
        ('safe', 1.1),
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        'full_body'
    ], ['nsfw', *generic_neg_words], 42


def _portrait_words(name):
    return [
        ('safe', 1.1),
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        'portrait',
        'looking_at_viewer',
    ], ['nsfw', *generic_neg_words], 42


def _profile_words(name):
    return [
        ('safe', 1.1),
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        'profile',
        'from_side',
        'upper_body',
    ], ['nsfw', *generic_neg_words], 42


def _free_words(name):
    return [
        ('safe', 1.1),
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
    ], ['nsfw', *generic_neg_words], 42


def _shorts_words(name):
    return [
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        'bandeau', 'belt', 'coat', 'cowboy_shot', 'looking_at_viewer', 'midriff', 'navel', 'open_clothes',
        'open_coat', 'short_shorts', 'shorts', 'smile', 'standing', 'stomach', 'strapless', 'long_sleeves',
        'tube_top', 'wide_sleeves', ':d', 'crop_top', 'open_mouth',
        'hand_on_hip', 'hand_up', 'simple_background', 'thighs', 'cleavage', 'jacket'
    ], generic_neg_words, 42


def _china_dress_words(name):
    return [
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        'bare_shoulders', 'bead_bracelet', 'beads', 'china_dress', 'chinese_clothes', 'dress',
        'jewelry', 'looking_at_viewer', 'sleeveless',
        'sleeveless_dress', 'solo', 'bracelet', 'smile', 'medium_breasts',
        'thighs', 'cowboy_shot', ':d', 'open_mouth'
    ], generic_neg_words, 42


def _bikini_words(name):
    return [
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        ('night', 1.1),
        ('starry sky', 1.1),
        'beach',
        'beautiful detailed sky',
        ('extremely detailed background', 1.2),
        ('standing', 1.1),
        'looking at viewer',
        ('bikini', 1.3),
        'light smile',
    ], generic_neg_words, 758691538


def _nude_words(name):
    return [
        'nsfw',
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        ('lying on bed', 1.1),
        ('extremely detailed background', 1.2),
        ('nude', 1.4),
        ('spread legs', 1.1),
        ('arms up', 1.1),
        'mature',
        'nipples',
        ('pussy', 1.15),
        ('pussy juice', 1.3),
        'looking at viewer',
        ('embarrassed', 1.1),
        'endured face',
        'feet out of frame',
    ], generic_neg_words, 465191133


def _nude_stand_words(name):
    return [
        'nsfw',
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        ('simple background', 1.1),
        ('standing', 1.15),
        ('nude', 1.4),
        ('completely nude', 1.2),
        'mature',
        'nipples',
        ('pussy', 1.15),
        ('pussy juice', 1.3),
        'looking at viewer',
        ('embarrassed', 1.1),
    ], generic_neg_words, 758691538


def _safe_maid_words(name):
    return [
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        ('maid', 1.4),
        ('long maid dress', 1.15),
    ], [
        'nsfw', 'sexy', 'underwear', 'bra', 'fishnet',
        'skin of legs', 'bare legs', 'bare skin', 'navel',
        *generic_neg_words,
    ], 42


def _safe_yukata_words(name):
    return [
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        ('yukata', 1.3),
        ('kimono', 1.15),
    ], [
        'nsfw', 'sexy', 'underwear', 'bra', 'fishnet',
        'skin of legs', 'bare legs', 'bare skin', 'navel',
        *generic_neg_words,
    ], 42


def _safe_miko_words(name):
    return [
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        ('white kimono', 1.35),
        ('red hakama', 1.35),
        ('wide sleeves', 1.2),
    ], [
        'nsfw', 'sexy', 'underwear', 'bra', 'fishnet',
        'skin of legs', 'bare legs', 'bare skin', 'navel',
        *generic_neg_words,
    ], 42


def _safe_suit_words(name):
    return [
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        ('black business suit', 1.4),
        ('tie', 1.2),
        ('sunglasses', 1.25),
        ('white gloves', 1.15),
        ('white shirt', 1.1),
        ('black skirt', 1.15),
        ('smoking', 1.2),
        'handsome',
    ], [
        'nsfw', 'sexy', 'underwear', 'bra', 'fishnet',
        'skin of legs', 'bare legs', 'bare skin', 'navel',
        *generic_neg_words,
    ], 42


def _sit_words(name):
    return [
        ('safe', 1.1),
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        'sitting',
        'sitting on chair',
        'chair',
        'cowboy_shot',
        'looking at viewer'
    ], ['nsfw', *generic_neg_words], 42


def _squat_words(name):
    return [
        ('safe', 1.1),
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        'squatting',
        'cowboy_shot',
        'looking at viewer'
    ], ['nsfw', *generic_neg_words], 42


def _kneel_words(name):
    return [
        ('safe', 1.1),
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        'kneeling',
        'kneeling on one knee',
        'on one knee',
        'cowboy_shot',
        'looking at viewer',
    ], ['nsfw', *generic_neg_words], 42


def _jump_words(name):
    return [
        ('safe', 1.1),
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        'jumping',
        'cowboy_shot',
        'looking at viewer',
    ], ['nsfw', *generic_neg_words], 42


def _crossed_arms_words(name):
    return [
        ('safe', 1.1),
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        'crossed_arms',
        'cowboy_shot',
        'looking at viewer',
    ], ['nsfw', *generic_neg_words], 42


def _angry_words(name):
    return [
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        'angry',
        'annoyed',
        'portrait',
        'looking at viewer',
    ], generic_neg_words, 42


def _smile_words(name):
    return [
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        'smile',
        'happy',
        'one_eye_closed',
        'portrait',
        'looking at viewer',
    ], generic_neg_words, 42


def _cry_words(name):
    return [
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        'crying',
        'sobbing',
        'tears',
        'portrait',
        'looking at viewer',
    ], generic_neg_words, 42


def _grin_words(name):
    return [
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        'evil_grin',
        'evil_smile',
        'grin',
        'portrait',
        'looking at viewer',
    ], generic_neg_words, 42


def _sex_words(name):
    return [
        'nsfw',
        *generic_words,
        (name, _DEFAULT_NAME_WEIGHT),
        '1boy', '1girl', 'shy', 'embarrassed',
        ('pussy', 1.25), 'penis', ('sex', 1.1), ('nude', 1.3), ('complete nude', 1.2),
        'breasts', ('nipples', 1.25), 'vagina', 'clitoris', 'pussy juice', ('cum in pussy', 1.1),
    ], generic_neg_words, 42


EXTRAS = [
    # basic
    ('portrait', _portrait_words, 3),
    ('full_body', _full_body_words, 2),
    ('profile', _profile_words, 2),
    ('free', _free_words, 2),

    # clothes
    ('shorts', _shorts_words, 1),
    ('maid', _safe_maid_words, 2),
    ('miko', _safe_miko_words, 1),
    ('yukata', _safe_yukata_words, 1),
    ('suit', _safe_suit_words, 1),
    ('china', _china_dress_words, 1),
    ('bikini', _bikini_words, 3),

    # posture
    ('sit', _sit_words, 1),
    ('squat', _squat_words, 1),
    ('kneel', _kneel_words, 1),
    ('jump', _jump_words, 1),
    ('crossed_arms', _crossed_arms_words, 1),

    # face
    ('angry', _angry_words, 1),
    ('smile', _smile_words, 1),
    ('cry', _cry_words, 1),
    ('grin', _grin_words, 1),

    # nsfw
    ('n_lie', _nude_words, 2),
    ('n_stand', _nude_stand_words, 3),
    ('n_sex', _sex_words, 2),
]


def _random_seed():
    return random.randint(0, 1 << 31)


def save_recommended_tags(character_name: str, clusters, workdir: str, base: int = 5, scale: int = 2):
    tags_dir = os.path.join(workdir, 'rtags')
    os.makedirs(tags_dir, exist_ok=True)

    def _yielder() -> Iterator[Tuple[str, str, str, int]]:
        for item in clusters:
            base_name = f'pattern_{item["id"]}'
            pos_words = [*generic_words, (character_name, _DEFAULT_NAME_WEIGHT), *item['tags']]
            pos_prompt_ = repr_tags(pos_words)
            neg_prompt_ = repr_tags(generic_neg_words)

            cnt = int(min(max(math.log(item['size'] / base) / math.log(scale) + 1, 1), 3))
            if cnt > 1:
                for i in range(cnt):
                    yield f'{base_name}_{i}', pos_prompt_, neg_prompt_, _random_seed()
            else:
                yield base_name, pos_prompt_, neg_prompt_, _random_seed()

        for name_, func_, repeats in EXTRAS:
            pos_words, neg_words, seed_ = func_(character_name)
            pos_prompt_ = repr_tags(pos_words)
            neg_prompt_ = repr_tags(neg_words)
            if seed_ is None:
                seed_ = _random_seed()

            if repeats > 1:
                for i in range(repeats):
                    yield f'{name_}_{i}', pos_prompt_, neg_prompt_, (seed_ if i == 0 else _random_seed())
            else:
                yield name_, pos_prompt_, neg_prompt_, seed_

    for idx, (name, pos_prompt, neg_prompt, seed) in enumerate(_yielder()):
        with open(os.path.join(tags_dir, f'{name}.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'index': idx,
                'name': name,
                'prompt': pos_prompt,
                'neg_prompt': neg_prompt,
                'seed': seed,
            }, f, indent=4, ensure_ascii=False)
