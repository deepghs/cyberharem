import glob
import io
import json
import logging
import os
import re
import textwrap
from typing import Union, Optional, List

import markdown2
import numpy as np
from PIL import Image
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from imgutils.data import load_image
from imgutils.detect import detect_faces
from imgutils.metrics import ccip_extract_feature, ccip_batch_differences, ccip_default_threshold
from imgutils.validate import anime_rating_score
from pycivitai import civitai_find_online
from pycivitai.client import find_version_id_by_hash
from tqdm.auto import tqdm
from waifuc.source import LocalSource

from .export import draw_with_repo
from ..dataset import load_dataset_for_character
from ..publish.civitai import _tag_decode, try_find_title, try_get_title_from_repo
from ..utils import srequest, get_hf_fs, load_tags_from_directory


def publish_samples_to_civitai(images_dir, model: Union[int, str], model_version: Optional[str] = None,
                               model_creator='narugo1992', safe_only: bool = False,
                               extra_tags: Optional[List[str]] = None, post_title: str = None,
                               session_repo: str = 'narugo/civitai_session_p1'):
    resource = civitai_find_online(model, model_version, creator=model_creator)
    model_version_id = resource.version_id
    post_title = post_title or f"{resource.model_name} - {resource.version_name} Review"

    images = []
    for img_file in glob.glob(os.path.join(images_dir, '*.png')):
        img_filename = os.path.basename(img_file)
        img_name = os.path.splitext(img_filename)[0]
        img_info_filename = f'{img_name}_info.txt'

        local_img_file = os.path.join(images_dir, img_filename)
        local_info_file = os.path.join(images_dir, img_info_filename)

        info = {}
        with open(local_info_file, 'r', encoding='utf-8') as iif:
            for line in iif:
                line = line.strip()
                if line:
                    info_name, info_text = line.split(':', maxsplit=1)
                    info[info_name.strip()] = info_text.strip()

        meta = {
            'cfgScale': int(round(float(info.get('Guidance Scale')))),
            'negativePrompt': info.get('Neg Prompt'),
            'prompt': info.get('Prompt'),
            'sampler': info.get('Sample Method', "Euler a"),
            'seed': int(info.get('Seed')),
            'steps': int(info.get('Infer Steps')),
            'Size': f"{info['Width']}x{info['Height']}",
        }
        if info.get('Clip Skip'):
            meta['clipSkip'] = int(info['Clip Skip'])
        if info.get('Model'):
            meta['Model'] = info['Model']
            pil_img_file = Image.open(local_img_file)
            if pil_img_file.info.get('parameters'):
                png_info_text = pil_img_file.info['parameters']
                find_hash = re.findall(r'Model hash:\s*([a-zA-Z\d]+)', png_info_text, re.IGNORECASE)
                if find_hash:
                    model_hash = find_hash[0].lower()
                    meta['hashes'] = {"model": model_hash}
                    meta["resources"] = [
                        {
                            "hash": model_hash,
                            "name": info['Model'],
                            "type": "model"
                        }
                    ]
                    meta["Model hash"] = model_hash

        nsfw = (info.get('Safe For Word', info.get('Safe For Work')) or '').lower() != 'yes'

        rating_score = anime_rating_score(local_img_file)
        safe_v = int(round(rating_score['safe'] * 10))
        safe_r15 = int(round(rating_score['r15'] * 10))
        safe_r18 = int(round(rating_score['r18'] * 10))
        faces = detect_faces(local_img_file)
        if faces:
            (x0, y0, x1, y1), _, _ = faces[0]
            width, height = load_image(local_img_file).size
            face_area = abs((x1 - x0) * (y1 - y0))
            face_ratio = face_area * 1.0 / (width * height)
            face_ratio = int(round(face_ratio * 50))
        else:
            continue

        images.append((
            (-safe_v, -safe_r15, -safe_r18) if safe_only else (0,),
            -face_ratio,
            1 if nsfw else 0,
            0 if img_name.startswith('pattern_') else 1,
            img_name,
            (local_img_file, img_filename, meta)
        ))

    images = [item[-1] for item in sorted(images)]

    from ..publish.civitai import civitai_upload_images, get_civitai_session, parse_publish_at

    def _custom_pc_func(mvid):
        return {
            "json": {
                "modelVersionId": mvid,
                "title": post_title,
                "tag": None,
                "authed": True,
            },
            "meta": {
                "values": {
                    "tag": ["undefined"]
                }
            }
        }

    session = get_civitai_session(session_repo)
    post_id = civitai_upload_images(
        model_version_id, images,
        tags=[*resource.tags, *extra_tags],
        model_id=resource.model_id,
        pc_func=_custom_pc_func,
        session=session,
    )

    logging.info(f'Publishing post {post_id!r} ...')
    resp = srequest(
        session, 'POST', 'https://civitai.com/api/trpc/post.update',
        json={
            "json": {
                "id": post_id,
                "publishedAt": parse_publish_at('now'),
                "authed": True,
            },
            "meta": {
                "values": {
                    "publishedAt": ["Date"]
                }
            }
        },
        headers={'Referer': f'https://civitai.com/models/{resource.model_id}/wizard?step=4'},
    )
    resp.raise_for_status()

    return images


def civitai_review(model: Union[int, str], model_version: Optional[str] = None,
                   model_creator='narugo1992', rating: int = 5, description_md: Optional[str] = None,
                   session_repo: str = 'narugo/civitai_session_p1'):
    resource = civitai_find_online(model, model_version, creator=model_creator)

    from ..publish.civitai import get_civitai_session
    session = get_civitai_session(session_repo)

    logging.info(f'Try find exist review of model version #{resource.version_id} ...')
    _err = None
    try:  # Add this shit for the 500 of this API (2023-09-14)
        resp = srequest(
            session, 'GET', 'https://civitai.com/api/trpc/resourceReview.getUserResourceReview',
            params={'input': json.dumps({"json": {"modelVersionId": resource.version_id, "authed": True}})},
            headers={
                'Referer': f'https://civitai.com/posts/create?modelId={resource.model_id}&'
                           f'modelVersionId={resource.version_id}&'
                           f'returnUrl=/models/{resource.model_id}?'
                           f'modelVersionId={resource.version_id}reviewing=true'
            },
            raise_for_status=False
        )
    except AssertionError:
        _err = True
        resp = None

    if _err or resp.status_code == 404:
        logging.info(f'Creating review for #{resource.version_id} ...')
        resp = srequest(
            session, 'POST', 'https://civitai.com/api/trpc/resourceReview.create',
            json={
                "json": {
                    "modelVersionId": resource.version_id,
                    "modelId": resource.model_id,
                    "rating": rating,
                    "authed": True,
                }
            },
            headers={'Referer': f'https://civitai.com/models/{resource.model_id}/wizard?step=4'}
        )
        resp.raise_for_status()
    else:
        if resp is not None:
            resp.raise_for_status()
    review_id = resp.json()['result']['data']['json']['id']

    logging.info(f'Updating review #{review_id}\'s rating ...')
    resp = srequest(
        session, 'POST', 'https://civitai.com/api/trpc/resourceReview.update',
        json={
            "json": {
                "id": review_id,
                "rating": rating,
                "details": None,
                "authed": True,
            },
            "meta": {"values": {"details": ["undefined"]}}
        },
        headers={'Referer': f'https://civitai.com/models/{resource.model_id}/wizard?step=4'}
    )
    resp.raise_for_status()

    if description_md:
        logging.info(f'Updating review #{review_id}\'s description ...')
        resp = srequest(
            session, 'POST', 'https://civitai.com/api/trpc/resourceReview.update',
            json={
                "json": {
                    "id": review_id,
                    "details": markdown2.markdown(textwrap.dedent(description_md)),
                    'rating': None,
                    "authed": True,
                },
                "meta": {"values": {"rating": ["undefined"]}}
            },
            headers={'Referer': f'https://civitai.com/models/{resource.model_id}/wizard?step=4'}
        )
        resp.raise_for_status()


_BASE_MODEL_LIST = [
    'AIARTCHAN/anidosmixV2',
    # 'stablediffusionapi/anything-v5',
    # 'Lykon/DreamShaper',
    'Meina/Unreal_V4.1',
    'digiplay/majicMIX_realistic_v6',
    'jzli/XXMix_9realistic-v4',
    'stablediffusionapi/abyssorangemix2nsfw',
    'AIARTCHAN/expmixLine_v2',
    # 'Yntec/CuteYuki2',
    'stablediffusionapi/counterfeit-v30',
    'stablediffusionapi/flat-2d-animerge',
    'redstonehero/cetusmix_v4',
    # 'KBlueLeaf/kohaku-v4-rev1.2',
    # 'stablediffusionapi/night-sky-yozora-sty',
    'Meina/MeinaHentai_V4',
    # 'Meina/MeinaPastel_V6',
]


def civitai_auto_review(repository: str, model: Optional[Union[int, str]] = None,
                        model_version: Optional[str] = None,
                        model_creator='narugo1992', step: Optional[int] = None,
                        base_models: Optional[List[str]] = None,
                        rating: Optional[int] = 5, description_md: Optional[str] = None,
                        session_repo: str = 'narugo/civitai_session_p1'):
    game_name = repository.split('/')[-1].split('_')[-1]
    char_name = ' '.join(repository.split('/')[-1].split('_')[:-1])
    model = model or try_find_title(char_name, game_name) or \
            try_get_title_from_repo(repository) or repository.split('/')[-1]
    logging.info(f'Model name on civitai: {model!r}')

    from ..publish.export import KNOWN_MODEL_HASHES

    hf_fs = get_hf_fs()
    model_info = json.loads(hf_fs.read_text(f'{repository}/meta.json'))
    dataset_info = model_info['dataset']

    # load dataset
    ds_size = (384, 512) if not dataset_info or not dataset_info['type'] else dataset_info['type']
    with load_dataset_for_character(repository, size=ds_size) as (_, ds_dir):
        core_tags, _ = load_tags_from_directory(ds_dir)

        all_tags = [
            game_name, f"{game_name} {char_name}", char_name,
            'female', 'girl', 'character', 'fully-automated', 'random prompt', 'random seed',
            *map(_tag_decode, core_tags.keys()),
        ]
        ds_source = LocalSource(ds_dir)
        ds_feats = []
        for item in tqdm(list(ds_source), desc='Extract Dataset Feature'):
            ds_feats.append(ccip_extract_feature(item.image))

        all_feats = []
        model_results = []
        for base_model in (base_models or _BASE_MODEL_LIST):
            logging.info(f'Reviewing with {base_model!r} ...')
            with TemporaryDirectory() as td:
                if KNOWN_MODEL_HASHES.get(base_model):
                    bm_id, bm_version_id, _ = find_version_id_by_hash(KNOWN_MODEL_HASHES[base_model])
                    resource = civitai_find_online(bm_id, bm_version_id)
                    m_name = f'{resource.model_name} - {resource.version_name}'
                    m_url = f'https://civitai.com/models/{resource.model_id}?modelVersionId={resource.version_id}'
                else:
                    m_name = base_model
                    m_url = None

                draw_with_repo(repository, td, step=step, pretrained_model=base_model)
                images = publish_samples_to_civitai(
                    td, model, model_version,
                    model_creator=model_creator,
                    extra_tags=all_tags,
                    post_title=f"AI Review (Base Model: {m_name})",
                    session_repo=session_repo
                )

                images_count = len(images)
                gp_feats = []
                for local_imgfile, _, _ in tqdm(images, desc='Extract Images Feature'):
                    gp_feats.append(ccip_extract_feature(local_imgfile))
                all_feats.extend(gp_feats)

                gp_diffs = ccip_batch_differences([*gp_feats, *ds_feats])[:len(gp_feats), len(gp_feats):]
                gp_batch = gp_diffs <= ccip_default_threshold()
                scores = gp_batch.mean(axis=1)
                losses = gp_diffs.mean(axis=1)

                ret = {
                    'model_name': m_name,
                    'model_homepage': m_url,
                    'images': images_count,
                    'mean_score': scores.mean().item(),
                    'median_score': np.median(scores).item(),
                    'mean_loss': losses.mean().item(),
                    'median_loss': np.median(losses).item(),
                }
                logging.info(f'Result of model: {ret!r}')
                model_results.append(ret)

        all_diffs = ccip_batch_differences([*all_feats, *ds_feats])[:len(all_feats), len(all_feats):]
        all_batch = all_diffs <= ccip_default_threshold()
        all_scores = all_batch.mean(axis=1)
        all_losses = all_diffs.mean(axis=1)
        all_mean_score = all_scores.mean().item()
        all_median_score = np.median(all_scores).item()
        all_mean_loss = all_losses.mean().item()
        all_median_loss = np.median(all_losses).item()

        if rating is not None:
            logging.info('Making review ...')
            with io.StringIO() as ds:
                print('Tested on the following models:', file=ds)
                print('', file=ds)

                all_total_images = 0
                for mr in model_results:
                    if mr['model_homepage']:
                        mx = f'[{mr["model_name"]}]({mr["model_homepage"]})'
                    else:
                        mx = mr['model_name']

                    all_total_images += mr['images']
                    print(
                        f'When using model {mx}, {plural_word(mr["images"], "image")} in total, '
                        f'recognition score (mean/median): {mr["mean_score"]:.3f}/{mr["median_score"]:.3f}, '
                        f'character image loss (mean/median): {mr["mean_loss"]:.4f}/{mr["median_loss"]:.4f}.',
                        file=ds
                    )
                    print('', file=ds)

                print(
                    f'Overall, {plural_word(all_total_images, "image")} in total, '
                    f'recognition score (mean/median): {all_mean_score:.3f}/{all_median_score:.3f}, '
                    f'character image loss (mean/median): {all_mean_loss:.4f}/{all_median_loss:.4f}.',
                    file=ds
                )
                print('', file=ds)

                description_md = description_md or ds.getvalue()

            try:
                civitai_review(model, model_version, model_creator, rating, description_md, session_repo)
            except:
                print('This is the description md:')
                print(description_md)
                raise
