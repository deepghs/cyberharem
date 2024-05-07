import logging
import os.path
import shutil
from typing import Optional, Union

from waifuc.action import CharacterEnhanceAction
from waifuc.source import BaseDataSource, LocalSource

from ..utils import get_global_namespace, get_hf_client


def train_iter(
        origin_source: Union[str, BaseDataSource], name: str, display_name: str,
        repository: Optional[str] = None, revision: str = 'main', workdir: Optional[str] = None,
        template_file: str = 'ch_lora_sd15.toml', pretrained_model: str = None,
        seed: int = None, use_reg: Optional[bool] = True, latent_cache_id: Optional[str] = None,
        bs: int = 8, unet_lr: float = 0.0006, te_lr: float = 0.0006, train_te: bool = False,
        dim: Optional[int] = None, alpha: int = 2, resolution: int = 720, res_ratio: float = 2.2,
        bangumi_style_tag: str = 'anime_style', comment: str = None, force_retrain: bool = False,
        tiny_scale: Optional[float] = 0.5, min_resolution: int = 720, train_rounds: int = 5,
        pattern_top_n: int = 1, top_n: int = 30, fidelity_alpha: float = 2.0,
        round_image_init_weight: float = 0.95, round_image_weight_decrease: float = 0.7,
        discord_publish: bool = True, origin_enhance: bool = True, origin_enhance_repeats: int = 25,
):
    workdir = workdir or os.path.join('runs', name)
    hf_client = get_hf_client()

    if isinstance(origin_source, str):
        origin_source = LocalSource(origin_source)
    elif isinstance(origin_source, BaseDataSource):
        pass
    else:
        raise TypeError(f'Unknown origin_source type, str or BaseDataSource expected but {origin_source!r} found.')
    if origin_enhance:
        origin_source = origin_source.attach(
            CharacterEnhanceAction(repeats=origin_enhance_repeats)
        )

    repository = repository or f'{get_global_namespace()}/{name}'
    logging.info(f'Repository: {repository!r}, name: {name!r}, display_name: {display_name!r}.')

    from ..dataset.crawler import crawl_dataset_to_huggingface

    for round_id in range(train_rounds):
        logging.info(f'------------- Round #{round_id} -------------')
        round_workdir = os.path.join(workdir, f'round_{round_id}')
        os.makedirs(round_workdir, exist_ok=True)
        round_revision = f'{revision}-r{round_id}'
        trained_flag_file = os.path.join(round_workdir, '.trained')
        if os.path.exists(trained_flag_file):
            logging.info(f'Round #{round_id} already trained, skipped.')
            continue

        if round_id == 0:
            logging.info('Making original dataset ...')
            crawl_dataset_to_huggingface(
                source=origin_source,
                name=name,
                display_name=display_name,
                repository=repository,
                tiny_scale=tiny_scale,
                min_resolution=min_resolution,
                revision=revision,
                discord_publish=discord_publish,
            )
        else:
            last_round_workdir = os.path.join(workdir, f'round_{round_id - 1}')

            from ..infer import infer_for_scale
            logging.info('Inferring for scales ...')
            infer_for_scale(last_round_workdir, infer_seed_count=8, max_n_steps=3)

            logging.info('Eval and select best images ...')
            from ..eval.infer import eval_for_infer_raw
            eval_for_infer_raw(last_round_workdir, pattern_top_n, top_n, fidelity_alpha)
            last_infer_selected = os.path.join(last_round_workdir, 'infer', 'selected')

            logging.info(f'Try making dataset for round #{round_id}')
            crawl_dataset_to_huggingface(
                source=LocalSource(last_infer_selected),
                name=name,
                display_name=display_name,
                repository=repository,
                tiny_scale=tiny_scale,
                min_resolution=min_resolution,
                revision=round_revision,
                discord_publish=False,
            )

            logging.info('Copying last features file ...')
            shutil.copy(
                os.path.join(last_round_workdir, 'features.npy'),
                os.path.join(round_workdir, 'features.npy'),
            )

        from .train import train_lora
        pre_rounds = list(range(round_id, 0, -1))
        train_lora(
            ds_repo_id=repository,
            dataset_name='stage3-p180-1200',
            workdir=round_workdir,
            template_file=template_file,
            pretrained_model=pretrained_model,
            seed=seed,
            use_reg=use_reg,
            latent_cache_id=latent_cache_id,
            bs=bs,
            unet_lr=unet_lr,
            te_lr=te_lr,
            train_te=train_te,
            dim=dim,
            alpha=alpha,
            resolution=resolution,
            res_ratio=res_ratio,
            bangumi_style_tag=bangumi_style_tag,
            comment=comment,
            force_retrain=force_retrain,
            eps=10,
            save_interval=1,
            ds_attach_revisions=[f'r{r}' for r in pre_rounds],
            group_weights={
                f'r{r}': round_image_init_weight * (round_image_weight_decrease ** ir)
                for ir, r in enumerate(pre_rounds)
            },
            group_attached_tags={
                f'r{r}': ['reference']
                for ir, r in enumerate(pre_rounds)
            }
        )

        from ..publish import deploy_to_huggingface
        logging.info('Deploy to huggingface ...')
        deploy_to_huggingface(
            workdir=round_workdir,
            repository=repository,
            ccip_check=None,
            discord_publish=discord_publish,
            revision=revision,
        )
        logging.info(f'Backup for round #{round_id} ...')
        if hf_client.revision_exists(repo_id=repository, repo_type='model', revision=round_revision):
            hf_client.delete_branch(repo_id=repository, repo_type='model', revision=round_revision)
        hf_client.create_branch(repo_id=repository, repo_type='model', branch=round_revision, revision=revision)
