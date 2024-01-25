import os
from typing import Optional

from ditk import logging
from waifuc.source import DanbooruSource, AnimePicturesSource, ZerochanSource, EmptySource

from cyberharem.utils import get_global_namespace
from .crawler import crawl_dataset_to_huggingface

logging.try_init_root(logging.INFO)
logger = logging.getLogger("pyrate_limiter")
logger.disabled = True


def crawl_with_tags(name: str, display_name: str, repo_id: str, repo_type: str = 'dataset', revision: str = 'main',
                    ap_tag: Optional[str] = None, zc_tag: Optional[str] = None, db_tag: Optional[str] = None,
                    drop_multi: bool = False, limit: Optional[int] = 500, private: bool = False):
    stage1, stage2 = EmptySource(), EmptySource()
    if ap_tag:
        stage1 = stage1 | AnimePicturesSource([ap_tag, 'solo'], select='original')
        if not drop_multi:
            stage2 = stage2 | AnimePicturesSource([ap_tag], select='original')
    if zc_tag:
        stage1 = stage1 | ZerochanSource(zc_tag, strict=True, select='full')
        if not drop_multi:
            stage2 = stage2 | ZerochanSource(zc_tag, select='full')
    if db_tag:
        if drop_multi:
            stage2 = stage2 | DanbooruSource([db_tag, 'solo'], min_size=3000)
        else:
            stage2 = stage2 | DanbooruSource([db_tag], min_size=3000)

    source = stage1 + stage2

    crawl_dataset_to_huggingface(
        source=source,
        repository=repo_id,
        repo_type=repo_type,
        revision=revision,
        name=name,
        display_name=display_name,
        limit=limit or None,
        drop_multi=drop_multi,
        private=private,
    )


if __name__ == '__main__':
    ch_limit = os.environ.get('CH_LIMIT')
    ch_limit = int(ch_limit) if ch_limit else None
    crawl_with_tags(
        name=os.environ['CH_NAME'],
        display_name=os.environ['CH_DISPLAY_NAME'],
        repo_id=os.environ['CH_REPO_ID'] or f'{get_global_namespace()}/{os.environ["CH_NAME"]}',
        ap_tag=os.environ.get('CH_AP_TAG'),
        zc_tag=os.environ.get('CH_ZC_TAG'),
        db_tag=os.environ.get('CH_DB_TAG'),
        drop_multi=bool(os.environ.get('CH_DROP_MULTI')),
        limit=ch_limit,
        private=bool(os.environ.get('CH_PRIVATE')),
    )
