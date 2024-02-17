import os

from ditk import logging
from gchar.games import get_character_class
from gchar.games.base import Character
from gchar.generic import import_generic

from .crawler import crawl_dataset_to_huggingface

import_generic()

logging.try_init_root(logging.INFO)
logger = logging.getLogger("pyrate_limiter")
logger.disabled = True

if __name__ == '__main__':
    game_cls = get_character_class(os.environ['CH_GAME_NAME'])
    ch: Character = game_cls.get(os.environ['CH_WAIFU_NAME'])
    crawl_dataset_to_huggingface(
        ch,
        limit=500,
        drop_multi=bool(os.environ.get('CH_DROP_MULTI')),
        remove_empty_repo=False,
        min_images=5,
    )
