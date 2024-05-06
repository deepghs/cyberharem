import logging
import os.path
from typing import Optional, Union

from waifuc.source import BaseDataSource, LocalSource

from ..utils import get_global_namespace


def train_iter(
        origin_source: Union[str, BaseDataSource], name: str, display_name: str,
        repository: Optional[str] = None, workdir: Optional[str] = None,
        template_file: str = 'ch_lora_sd15.toml', pretrained_model: str = None,
        seed: int = None, use_reg: Optional[bool] = False, latent_cache_id: Optional[str] = None,
        bs: int = 8, unet_lr: float = 0.0006, te_lr: float = 0.0006, train_te: bool = False,
        dim: Optional[int] = None, alpha: int = 2, resolution: int = 720, res_ratio: float = 2.2,
        bangumi_style_tag: str = 'anime_style', comment: str = None, force_retrain: bool = False,
        tiny_scale: Optional[float] = 0.5, min_resolution: int = 720, rounds: int=5
):
    workdir = os.path.join('runs', name)

    if isinstance(origin_source, str):
        origin_source = LocalSource(origin_source)
    elif isinstance(origin_source, BaseDataSource):
        pass
    else:
        raise TypeError(f'Unknown origin_source type, str or BaseDataSource expected but {origin_source!r} found.')

    repository = repository or f'{get_global_namespace()}/{name}'
    logging.info(f'Repository: {repository!r}, name: {name!r}, display_name: {display_name!r}.')

    from ..dataset.crawler import crawl_dataset_to_huggingface


    logging.info('Making original dataset ...')
    crawl_dataset_to_huggingface(
        source=origin_source,
        name=name,
        display_name=display_name,
        repository=repository,
        tiny_scale=tiny_scale,
        min_resolution=min_resolution,
        revision='main'
    )

    for round_id in range(rounds):
        round_workdir = os.path.join(workdir, f'round_{round_id}')
        if round_id:
            pass
        pass

