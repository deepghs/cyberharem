import re

from gchar.games.base import Character
from gchar.generic import import_generic
from thefuzz import fuzz

import_generic()


def get_pure_name(name: str) -> str:
    return '_'.join([word for word in re.split(r'[\W_]+', name.lower()) if word])


def get_alphabet_name(name: str) -> str:
    return '_'.join(re.findall(r'[a-zA-Z\d+]+', name.lower()))


def _name_alphabet_ratio(name: str) -> float:
    pure_name = get_pure_name(name)
    alphabet_name = get_alphabet_name(name)
    return fuzz.token_set_ratio(pure_name, alphabet_name)


def get_ch_name(ch: Character):
    names = [
        *map(str, ch.ennames),
        *map(str, ch.cnnames),
        *map(str, ch.jpnames),
    ]
    all_names = [(name, _name_alphabet_ratio(name), i) for i, name in enumerate(names)]
    all_names = sorted(all_names, key=lambda x: (-x[1], x[2]))

    name, ratio, _ = all_names[0]
    if ratio >= 0.9:
        short_name = get_alphabet_name(name)
    else:
        raise ValueError(f'No suitable alphabet-based name for {ch!r}.')

    return f'{short_name}_{ch.__game_name__}'


def get_formal_title(ch: Character):
    names = []
    if ch.enname:
        names.append(str(ch.enname))
    if ch.jpname:
        names.append(str(ch.jpname))
    if ch.cnname:
        names.append(str(ch.cnname))
    if hasattr(ch, 'krname') and ch.krname:
        names.append(str(ch.krname))

    return f"{'/'.join(names)} ({ch.__class__.__official_name__})"
