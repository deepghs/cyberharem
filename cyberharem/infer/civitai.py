import glob
import json
import logging
import os
import re
import textwrap
from typing import Union, Optional, List

import markdown
from PIL import Image
from hbutils.system import TemporaryDirectory
from imgutils.data import load_image
from imgutils.detect import detect_faces
from imgutils.validate import anime_rating_score
from pycivitai import civitai_find_online

from cyberharem.utils import srequest
from .export import draw_with_repo


def publish_samples_to_civitai(images_dir, model: Union[int, str], model_version: Optional[str] = None,
                               model_creator='narugo1992', safe_only: bool = False,
                               session_repo: str = 'narugo/civitai_session_p1'):
    resource = civitai_find_online(model, model_version, creator=model_creator)
    model_version_id = resource.version_id
    post_title = f"{resource.model_name} - {resource.version_name} Review"

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
        tags=resource.tags,
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


def civitai_review(model: Union[int, str], model_version: Optional[str] = None,
                   model_creator='narugo1992', rating: int = 5, description_md: Optional[str] = None,
                   session_repo: str = 'narugo/civitai_session_p1'):
    resource = civitai_find_online(model, model_version, creator=model_creator)

    from ..publish.civitai import get_civitai_session
    session = get_civitai_session(session_repo)

    logging.info('Try find exist review ...')
    resp = srequest(
        session, 'GET', 'https://civitai.com/api/trpc/resourceReview.getUserResourceReview',
        params={'input': json.dumps({"json": {"modelVersionId": resource.version_id, "authed": True}})},
        headers={'Referer': f'https://civitai.com/models/{resource.model_id}/wizard?step=2'},
        raise_for_status=False
    )
    if resp.status_code == 404:
        logging.info('Creating review ...')
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
        resp.raise_for_status()
    review_id = resp.json()['result']['data']['json']['id']

    logging.info(f'Updating review {review_id}\'s rating ...')
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
        logging.info(f'Updating review {review_id}\'s description ...')
        resp = srequest(
            session, 'POST', 'https://civitai.com/api/trpc/resourceReview.update',
            json={
                "json": {
                    "id": review_id,
                    "details": markdown.markdown(textwrap.dedent(description_md)),
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
    'Lykon/DreamShaper',
    # 'digiplay/majicMIX_realistic_v6',
    'jzli/XXMix_9realistic-v4',
    'stablediffusionapi/abyssorangemix2nsfw',
    'AIARTCHAN/expmixLine_v2',
    # 'Yntec/CuteYuki2',
    'stablediffusionapi/counterfeit-v30',
    'stablediffusionapi/flat-2d-animerge',
    'redstonehero/cetusmix_v4',
]


def civitai_auto_review(repository: str, model: Union[int, str], model_version: Optional[str] = None,
                        model_creator='narugo1992', step: Optional[int] = None,
                        base_models: Optional[List[str]] = None,
                        rating: Optional[int] = 5, description_md: Optional[str] = None,
                        session_repo: str = 'narugo/civitai_session_p1'):
    for base_model in (base_models or _BASE_MODEL_LIST):
        logging.info(f'Reviewing with {base_model!r} ...')
        with TemporaryDirectory() as td:
            draw_with_repo(repository, td, step=step, pretrained_model=base_model)
            publish_samples_to_civitai(td, model, model_version,
                                       model_creator=model_creator, session_repo=session_repo)

    if rating is not None:
        logging.info('Making review ...')
        civitai_review(model, model_version, model_creator, rating, description_md, session_repo)
