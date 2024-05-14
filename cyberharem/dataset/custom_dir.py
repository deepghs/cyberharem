import os
from typing import Optional

from ditk import logging
from hbutils.system import TemporaryDirectory
from hfutils.operate import download_archive_as_directory
from hfutils.utils import parse_hf_fs_path
from waifuc.source import LocalSource

from cyberharem.utils import get_global_namespace
from .crawler import crawl_dataset_to_huggingface

logging.try_init_root(logging.INFO)
logger = logging.getLogger("pyrate_limiter")
logger.disabled = True


def crawl_with_custom_hf_dir(name: str, display_name: str, repo_id: str, origin_package_path: str,
                             repo_type: str = 'dataset', revision: str = 'main',
                             drop_multi: bool = False, limit: Optional[int] = 500,
                             bangumi_source_repository: Optional[str] = None, private: bool = False):
    with TemporaryDirectory() as td:
        parsed = parse_hf_fs_path(origin_package_path)
        download_archive_as_directory(
            repo_id=parsed.repo_id,
            repo_type=parsed.repo_type,
            file_in_repo=parsed.filename,
            local_directory=td,
            revision=parsed.revision or 'main',
        )
        source = LocalSource(td, shuffle=True)

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
            bangumi_source_repository=bangumi_source_repository,
        )


if __name__ == '__main__':
    ch_limit = os.environ.get('CH_LIMIT')
    ch_limit = int(ch_limit) if ch_limit else None
    crawl_with_custom_hf_dir(
        name=os.environ['CH_NAME'],
        display_name=os.environ['CH_DISPLAY_NAME'],
        repo_id=os.environ['CH_REPO_ID'] or f'{get_global_namespace()}/{os.environ["CH_NAME"]}',
        origin_package_path=os.environ['CH_ORIGIN_PACKAGE'],
        drop_multi=bool(os.environ.get('CH_DROP_MULTI')),
        limit=ch_limit,
        private=bool(os.environ.get('CH_PRIVATE')),
        bangumi_source_repository=os.environ.get('CH_BANGUMI_REPO') or None,
    )
