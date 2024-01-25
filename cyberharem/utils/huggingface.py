import math
import os
from functools import partial

from huggingface_hub import configure_http_backend, HfApi, HfFileSystem

from .session import get_requests_session

_NUM_TAGS = [
    ('n<1K', 0, 1_000),
    ('1K<n<10K', 1_000, 10_000),
    ('10K<n<100K', 10_000, 100_000),
    ('100K<n<1M', 100_000, 1_000_000),
    ('1M<n<10M', 1_000_000, 10_000_000),
    ('10M<n<100M', 10_000_000, 100_000_000),
    ('100M<n<1B', 100_000_000, 1_000_000_000),
    ('1B<n<10B', 1_000_000_000, 10_000_000_000),
    ('10B<n<100B', 10_000_000_000, 100_000_000_000),
    ('100B<n<1T', 100_000_000_000, 1_000_000_000_000),
    ('n>1T', 1_000_000_000_000, math.inf),
]


def number_to_tag(v):
    for tag, min_, max_ in _NUM_TAGS:
        if min_ <= v < max_:
            return tag

    raise ValueError(f'No tags found for {v!r}')


configure_http_backend(partial(get_requests_session, timeout=120))


def get_hf_client() -> HfApi:
    return HfApi(token=os.environ.get('HF_TOKEN'))


def get_hf_fs() -> HfFileSystem:
    return HfFileSystem(token=os.environ.get('HF_TOKEN'), use_listings_cache=False)


_DEFAULT_NAMESPACE = 'CyberHarem'


def get_global_namespace() -> str:
    return os.environ.get('CH_NAMESPACE', _DEFAULT_NAMESPACE) or _DEFAULT_NAMESPACE
