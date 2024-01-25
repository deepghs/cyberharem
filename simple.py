from ditk import logging
from gchar.games.arknights import Character

from cyberharem.dataset import crawl_dataset_to_huggingface

logging.try_init_root(logging.INFO)
logger = logging.getLogger("pyrate_limiter")
logger.disabled = True

crawl_dataset_to_huggingface(
    Character.get('amiya'),
    repository='narugo/test_amiya_50_x_better',
    limit=50,
    private=True
)
