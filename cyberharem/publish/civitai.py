import glob
import json
import logging
import math
import os.path
import re
import textwrap
import uuid
from typing import Optional, Tuple, List, Union

import blurhash
import numpy as np
from PIL import Image
from gchar.games.base import Character
from gchar.games.dispatch.access import GAME_CHARS
from gchar.generic import import_generic
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from huggingface_hub import hf_hub_url
from imgutils.data import load_image
from imgutils.detect import detect_faces
from imgutils.metrics import ccip_extract_feature, ccip_batch_same
from imgutils.validate import anime_rating_score, nsfw_pred
from pycivitai import civitai_find_online
from pycivitai.client import ModelNotFound
from tqdm.auto import tqdm
from urlobject import URLObject
from waifuc.source import LocalSource

try:
    from typing import Literal
except (ModuleNotFoundError, ImportError):
    from typing_extensions import Literal

import markdown2

from ..utils import get_civitai_session, srequest, get_ch_name, get_hf_fs, download_file, parse_time, \
    load_tags_from_directory, repr_tags

import_generic()


def _norm(x, keep_space: bool = True):
    return re.sub(r'[\W_]+', ' ' if keep_space else '', x.lower()).strip()


def _model_tag_same(x, y):
    return _norm(x, keep_space=True) == _norm(y, keep_space=True)


def civitai_query_model_tags(tag: str, session=None) -> Tuple[Optional[int], str]:
    session = session or get_civitai_session()
    logging.info(f'Querying tag {tag!r} from civitai ...')
    resp = srequest(session, 'GET', 'https://civitai.com/api/trpc/tag.getAll', params={
        'input': json.dumps({
            "json": {
                "limit": 20,
                "entityType": ["Model"],
                "categories": False,
                "query": tag,
                "authed": True,
            }
        })
    }, headers={'Referer': 'https://civitai.com/models/create'})

    data = resp.json()['result']['data']['json']['items']
    for item in data:
        if _model_tag_same(item['name'], tag):
            logging.info(f'Tag {item["name"]}({item["id"]}) found on civitai.')
            return item['id'], item['name']
    else:
        logging.info(f'Tag not found on civitai, new tag {_norm(tag)!r} will be created.')
        return None, _norm(tag)


CommercialUseTyping = Literal['none', 'image', 'rentCivit', 'rent', 'sell']


def civitai_upsert_model(
        name, description_md: str, tags: List[str],
        commercial_use: CommercialUseTyping = 'rent',
        allow_no_credit: bool = True, allow_derivatives: bool = True, allow_different_licence: bool = True,
        nsfw: bool = False, poi: bool = False, exist_model_id: Optional[int] = None,
        session=None
) -> Tuple[int, bool]:
    session = session or get_civitai_session()
    _exist_tags, tag_list, _tag_id = set(), [], 0
    _meta_values = {}
    for tag in tags:
        tag_id, tag_name = civitai_query_model_tags(tag, session)
        if tag_name not in _exist_tags:
            tag_list.append({'id': tag_id, 'name': tag_name})
            _meta_values[f"tagsOnModels.{_tag_id}.id"] = ["undefined"]
            _tag_id += 1

    post_json = {
        "name": name,
        "description": markdown2.markdown(textwrap.dedent(description_md)),
        "type": "LORA",

        "allowCommercialUse": commercial_use.lower().capitalize(),  # None, Image, Rent, Sell
        "allowNoCredit": allow_no_credit,
        "allowDerivatives": allow_derivatives,
        "allowDifferentLicense": allow_different_licence,

        "nsfw": nsfw,
        "poi": poi,
        "tagsOnModels": tag_list,

        "authed": True,
        "status": "Draft",
        "checkpointType": None,
        "uploadType": "Created",
    }
    if exist_model_id:
        post_json['id'] = exist_model_id
        post_json["locked"] = False
        post_json["status"] = "Published"
        logging.info(f'Model {name!r}({exist_model_id}) already exist, updating its new information. '
                     f'Tags: {[item["name"] for item in tag_list]!r} ...')
    else:
        logging.info(f'Creating model {name!r}, tags: {[item["name"] for item in tag_list]!r} ...')

    resp = session.post(
        'https://civitai.com/api/trpc/model.upsert',
        json={
            "json": post_json,
            "meta": {
                "values": _meta_values,
            }
        },
        headers={'Referer': 'https://civitai.com/models/create'},
    )

    data = resp.json()['result']['data']['json']
    return data['id'], data['nsfw']


def civitai_query_vae_models(session=None, model_id=None):
    session = session or get_civitai_session()
    logging.info('Querying VAE models ...')
    resp = srequest(
        session, 'GET', ' https://civitai.com/api/trpc/modelVersion.getModelVersionsByModelType',
        params={'input': json.dumps({"json": {"type": "VAE", "authed": True}})},
        headers={'Referer': f'https://civitai.com/models/{model_id or 0}/wizard?step=2'}
    )

    data = resp.json()['result']['data']['json']
    logging.info(f'{plural_word(len(data), "VAE model")} found.')
    return data


def _vae_model_same(x, y):
    return _norm(x, keep_space=False) == _norm(y, keep_space=False)


def civitai_create_version(
        model_id: int, version_name: str, description_md: str, trigger_words: List[str],
        base_model: str = 'SD 1.5', steps: Optional[int] = None, epochs: Optional[int] = None,
        clip_skip: Optional[int] = 2, vae_name: Optional[str] = None, early_access_time: int = 0,
        session=None
):
    session = session or get_civitai_session()

    vae_id = None
    if vae_name:
        for vae_item in civitai_query_vae_models(session, model_id):
            if _vae_model_same(vae_item['modelName'], vae_name):
                vae_id = vae_item['id']

    logging.info(f'Creating version {version_name!r} for model {model_id}, with base model {base_model!r} ...')
    resp = srequest(
        session, 'POST', 'https://civitai.com/api/trpc/modelVersion.upsert',
        json={
            "json": {
                "modelId": model_id,
                "name": version_name,
                "baseModel": base_model,
                "description": markdown2.markdown(textwrap.dedent(description_md)),
                "steps": steps,
                "epochs": epochs,
                "clipSkip": clip_skip,
                "vaeId": vae_id,
                "trainedWords": trigger_words,
                "earlyAccessTimeFrame": early_access_time,
                "skipTrainedWords": bool(not trigger_words),
                "authed": True,
            }
        },
        headers={'Referer': f'https://civitai.com/models/{model_id}/wizard?step=2'}
    )

    return resp.json()['result']['data']['json']


def civitai_upload_file(local_file: str, type_: str = 'model', filename: str = None,
                        model_id: int = None, session=None):
    session = session or get_civitai_session()
    filename = filename or os.path.basename(local_file)

    logging.info(f'Creating uploading request for {filename!r} ...')
    resp = srequest(
        session, 'POST', 'https://civitai.com/api/upload',
        json={
            "filename": filename,
            "type": type_,
            "size": os.path.getsize(local_file),
        },
        headers={'Referer': f'https://civitai.com/models/{model_id or 0}/wizard?step=3'}
    )
    upload_data = resp.json()

    logging.info(f'Uploading file {local_file!r} as {filename!r} ...')
    with open(local_file, 'rb') as f:
        resp = srequest(
            session, 'PUT', upload_data['urls'][0]['url'], data=f,
            headers={'Referer': f'https://civitai.com/models/{model_id or 0}/wizard?step=3'},
        )
        etag = resp.headers['ETag']

    logging.info(f'Completing uploading for {filename!r} ...')
    resp = srequest(
        session, 'POST', 'https://civitai.com/api/upload/complete',
        json={
            "bucket": upload_data['bucket'],
            "key": upload_data['key'],
            "type": type_,
            "uploadId": upload_data['uploadId'],
            "parts": [{"ETag": etag, "PartNumber": 1}],
        },
        headers={'Referer': f'https://civitai.com/models/{model_id or 0}/wizard?step=3'},
    )
    resp.raise_for_status()

    return {
        "url": str(URLObject(upload_data['urls'][0]['url']).without_query()),
        "bucket": upload_data['bucket'],
        "key": upload_data['key'],
        "name": filename,
        "uuid": str(uuid.uuid4()),
        "sizeKB": os.path.getsize(local_file) / 1024.0,
    }


def civitai_upload_models(model_version_id: int, model_files: List[Union[str, Tuple[str, str]]],
                          model_id: int = None, session=None):
    session = session or get_civitai_session()
    file_items = []
    for file_item in model_files:
        if isinstance(file_item, str):
            local_file, filename = file_item, file_item
        elif isinstance(file_item, tuple):
            local_file, filename = file_item
        else:
            raise TypeError(f'Unknown file type - {file_item!r}.')
        file_items.append((local_file, filename))

    for local_file, filename in file_items:
        upload_data = civitai_upload_file(local_file, 'model', filename, model_id, session)
        logging.info(f'Creating {filename!r} as model file of version {model_version_id} ...')
        resp = srequest(
            session, 'POST', 'https://civitai.com/api/trpc/modelFile.create',
            json={
                'json': {
                    **upload_data,
                    "modelVersionId": model_version_id,
                    "type": "Model",
                    "metadata": {
                        "size": None,
                        "fp": None
                    },
                    "authed": True
                },
            },
            headers={'Referer': f'https://civitai.com/models/{model_id or 0}/wizard?step=3'},
        )
        resp.raise_for_status()


def civitai_get_model_info(model_id: int, session=None):
    session = session or get_civitai_session()
    resp = srequest(
        session, 'GET', 'https://civitai.com/api/trpc/model.getById',
        params={'input': json.dumps({"json": {"id": model_id, "authed": True}})},
        headers={'Referer': f'https://civitai.com/models/{model_id}/wizard?step=4'},
    )
    return resp.json()['result']['data']['json']


def get_clamped_size(width, height, max_val, _type='all'):
    if _type == 'all':
        if width >= height:
            _type = 'width'
        elif height >= width:
            _type = 'height'

    if _type == 'width' and width > max_val:
        return max_val, int(round((height / width) * max_val))

    if _type == 'height' and height > max_val:
        return int(round((width / height) * max_val)), max_val

    return width, height


def parse_publish_at(publish_at: Optional[str] = None, keep_none: bool = True) -> Optional[str]:
    try:
        from zoneinfo import ZoneInfo
    except (ImportError, ModuleNotFoundError):
        from backports.zoneinfo import ZoneInfo

    if not keep_none and publish_at is None:
        publish_at = 'now'
    if publish_at is not None:
        local_time = parse_time(publish_at)
        publish_at = local_time.astimezone(ZoneInfo('UTC')).isoformat()

    return publish_at


def _post_create_func(model_version_id):
    return {
        "json": {
            "modelVersionId": model_version_id,
            "authed": True,
        }
    }


def civitai_upload_images(
        model_version_id: int, image_files: List[Union[str, Tuple[str, str], Tuple[str, str, dict]]],
        tags: List[str], nsfw: bool = False, model_id: int = None, pc_func=_post_create_func, session=None
):
    session = session or get_civitai_session()

    image_items = []
    for image_item in image_files:
        if isinstance(image_item, str):
            local_file, filename, meta = image_item, image_item, {}
        elif isinstance(image_item, tuple):
            if len(image_item) == 2:
                (local_file, filename), meta = image_item, {}
            elif len(image_item) == 3:
                local_file, filename, meta = image_item
            else:
                raise ValueError(f'Invalid image file format - {image_item!r}.')
        else:
            raise TypeError(f'Invalid image file type - {image_item!r}.')
        image_items.append((local_file, filename, meta))

    logging.info(f'Creating post for model version {model_version_id} ...')
    resp = srequest(
        session, 'POST', 'https://civitai.com/api/trpc/post.create',
        json=pc_func(model_version_id),
        headers={'Referer': f'https://civitai.com/models/{model_id or 0}/wizard?step=4'},
    )
    post_id = resp.json()['result']['data']['json']['id']

    for index, (local_file, filename, meta) in enumerate(image_items):
        logging.info(f'Creating image uploading request for image {filename!r} ...')
        resp = srequest(
            session, 'POST', 'https://civitai.com/api/image-upload',
            json={
                "filename": filename,
                "metadata": {}
            },
            headers={'Referer': f'https://civitai.com/models/{model_id or 0}/wizard?step=4'},
        )
        upload_id = resp.json()['id']
        upload_url = resp.json()['uploadURL']

        logging.info(f'Uploading local image {local_file!r} as image {filename!r} ...')
        with open(local_file, 'rb') as f:
            resp = srequest(session, 'PUT', upload_url, data=f)
            resp.raise_for_status()

        img = load_image(local_file, force_background='white', mode='RGB')
        new_width, new_height = get_clamped_size(img.width, img.height, 32)
        bhash = blurhash.encode(np.array(img.resize((new_width, new_height))))
        logging.info(f'Completing the uploading of {filename!r} ...')
        resp = srequest(
            session, 'POST', 'https://civitai.com/api/trpc/post.addImage',
            json={
                "json": {
                    "type": "image",
                    "index": index,
                    "uuid": str(uuid.uuid4()),
                    "name": filename,
                    "meta": meta,
                    "url": upload_id,
                    "mimeType": "image/png",
                    "hash": bhash,
                    "width": img.width,
                    "height": img.height,
                    "status": "uploading",
                    "message": None,
                    "postId": post_id,
                    "modelVersionId": model_version_id,
                    "authed": True
                },
                "meta": {
                    "values": {
                        "message": [
                            "undefined"
                        ]
                    }
                }
            },
            headers={'Referer': f'https://civitai.com/models/{model_id or 0}/wizard?step=4'},
        )
        resp.raise_for_status()

    for tag in tags:
        tag_id, tag_name = civitai_query_model_tags(tag, session)
        if tag_id is not None:
            logging.info(f'Adding tag {tag_name!r}({tag_id}) for post {post_id!r} ...')
            resp = srequest(
                session, 'POST', 'https://civitai.com/api/trpc/post.addTag',
                json={
                    "json": {
                        "id": post_id,
                        "tagId": tag_id,
                        "name": tag_name,
                        "authed": True,
                    }
                },
                headers={'Referer': f'https://civitai.com/models/{model_id or 0}/wizard?step=4'},
            )
        else:
            logging.info(f'Creating and adding new tag {tag_name!r} for post {post_id!r} ...')
            resp = srequest(
                session, 'POST', 'https://civitai.com/api/trpc/post.addTag',
                json={
                    "json": {
                        "id": post_id,
                        "tagId": None,
                        "name": tag_name,
                        "authed": True,
                    },
                    "meta": {
                        "values": {
                            "tagId": ["undefined"]
                        }
                    }
                },
                headers={'Referer': f'https://civitai.com/models/{model_id or 0}/wizard?step=4'},
            )

        resp.raise_for_status()

    logging.info(f'Marking for nsfw ({nsfw!r}) ...')
    resp = srequest(
        session, 'POST', 'https://civitai.com/api/trpc/post.update',
        json={
            'json': {
                'id': post_id,
                'nsfw': nsfw,
                'authed': True,
            }
        },
        headers={'Referer': f'https://civitai.com/models/{model_id or 0}/wizard?step=4'},
    )
    resp.raise_for_status()

    return post_id


def civiti_publish(model_id: int, model_version_id: int, publish_at=None, session=None):
    session = session or get_civitai_session()
    publish_at = parse_publish_at(publish_at, keep_none=True)

    if publish_at:
        logging.info(f'Publishing model {model_id!r}\'s version {model_version_id!r}, at {publish_at!r} ...')
    else:
        logging.info(f'Publishing model {model_id!r}\'s version {model_version_id!r} ...')
    resp = srequest(
        session, 'POST', 'https://civitai.com/api/trpc/model.publish',
        json={
            "json": {
                "id": model_id,
                "versionIds": [
                    model_version_id
                ],
                "publishedAt": publish_at,
                "authed": True
            },
            "meta": {
                "values": {
                    "publishedAt": [
                        "undefined" if publish_at is None else "Date",
                    ]
                }
            }
        },
        headers={'Referer': f'https://civitai.com/models/{model_id or 0}/wizard?step=4'},
    )
    resp.raise_for_status()


def try_find_title(char_name, game_name):
    try:
        game_cls = GAME_CHARS[game_name.lower()]
        ch = game_cls.get(char_name)
        if ch:
            names = []
            if ch.enname:
                names.append(str(ch.enname))
            if ch.jpname:
                names.append(str(ch.jpname))
            if ch.cnname:
                names.append(str(ch.cnname))
            if hasattr(ch, 'krname') and ch.krname:
                names.append(str(ch.krname))

            return f"{'/'.join(names)} ({game_cls.__official_name__})"

        else:
            cname = ' '.join(list(map(str.capitalize, char_name.split(' '))))
            return f'{cname} ({game_cls.__official_name__})'

    except KeyError:
        return None


def try_get_title_from_repo(repo):
    hf_fs = get_hf_fs()
    print(f'datasets/{repo}/meta.json')
    if hf_fs.exists(f'datasets/{repo}/meta.json'):
        data = json.loads(hf_fs.read_text(f'datasets/{repo}/meta.json'))
        character_name = data['name']

        source_name = repo.split('_')[-1]
        if hf_fs.exists(f'datasets/BangumiBase/{source_name}/meta.json'):
            base_data = json.loads(hf_fs.read_text(f'datasets/BangumiBase/{source_name}/meta.json'))
            source_full_name = base_data['name']
            return f'{character_name} ({source_full_name})'
        else:
            return character_name
    else:
        return None


def _tag_decode(text):
    return re.sub(r'[\s_]+', ' ', re.sub(r'\\([\\()])', r'\1', text)).strip()


def civitai_publish_from_hf(source, model_name: str = None, model_desc_md: str = None,
                            version_name: Optional[str] = None, version_desc_md: str = None,
                            step: Optional[int] = None, epoch: Optional[int] = None, upload_min_epoch: int = 6,
                            draft: bool = False, publish_at=None, allow_nsfw_images: bool = True,
                            force_create_model: bool = False, no_ccip_check: bool = False, session=None):
    if isinstance(source, Character):
        repo = f'CyberHarem/{get_ch_name(source)}'
    elif isinstance(source, str):
        repo = source
    else:
        raise TypeError(f'Unknown source type - {source!r}.')
    hf_fs = get_hf_fs()
    meta_json = json.loads(hf_fs.read_text(f'{repo}/meta.json'))
    game_name = repo.split('_')[-1]

    dataset_info = meta_json.get('dataset')
    ds_size = (384, 512) if not dataset_info or not dataset_info['type'] else dataset_info['type']
    with load_dataset_for_character(repo, size=ds_size) as (_, d):
        if dataset_info and dataset_info['size']:
            dataset_size = dataset_info['size']
        else:
            dataset_size = len(glob.glob(os.path.join(d, '*.png')))
        core_tags, _ = load_tags_from_directory(d)
        logging.info(f'Size of dataset if {dataset_size!r}.')

        ccip_feats = []
        for item in tqdm(list(LocalSource(d)[:10]), desc='Extracting features'):
            ccip_feats.append(ccip_extract_feature(item.image))

    version_name = version_name or meta_json.get('mark') or 'v1.0'
    all_steps = meta_json['steps']
    logging.info(f'Available steps: {all_steps!r}.')
    if step is not None:
        if epoch is not None:
            logging.warning(f'Step {step!r} is set, epoch value ({epoch}) will be ignored.')
    else:
        if epoch is not None:
            step = dataset_size * epoch
        else:
            if 'best_step' in meta_json:
                if upload_min_epoch is not None:
                    upload_min_step = upload_min_epoch * dataset_size
                else:
                    upload_min_step = -1
                best_step, best_score = None, None
                for score_item in meta_json["scores"]:
                    if best_step is None or \
                            (score_item['step'] >= upload_min_step and score_item['score'] >= best_score):
                        best_step, best_score = score_item['step'], score_item['score']

                if best_step is not None:
                    step = best_step
                else:
                    step = meta_json['best_step']
            else:
                step = max(all_steps)

    logging.info(f'Expected step is {step!r}.')
    _, _actual_step = sorted([(abs(s - step), s) for s in all_steps])[0]
    if _actual_step != step:
        logging.info(f'Actual used step is {_actual_step!r}.')

    step = _actual_step
    epoch = int(math.ceil(step / dataset_size))
    logging.info(f'Using step {step}, epoch {epoch}.')

    with TemporaryDirectory() as td:
        models_dir = os.path.join(td, 'models')
        os.makedirs(models_dir, exist_ok=True)

        lora_file = os.path.basename(hf_fs.glob(f'{repo}/{step}/*.safetensors')[0])
        pt_file = os.path.basename(hf_fs.glob(f'{repo}/{step}/*.pt')[0])
        trigger_word = os.path.splitext(lora_file)[0]
        char_name = ' '.join(trigger_word.split('_')[:-1])

        models = []
        local_lora_file = os.path.join(models_dir, lora_file)
        download_file(hf_hub_url(repo, filename=f'{step}/{lora_file}'), local_lora_file)
        models.append((local_lora_file, lora_file))
        local_pt_file = os.path.join(models_dir, pt_file)
        download_file(hf_hub_url(repo, filename=f'{step}/{pt_file}'), local_pt_file)
        models.append((local_pt_file, pt_file))

        images_dir = os.path.join(td, 'images')
        os.makedirs(images_dir, exist_ok=True)

        images = []
        tags_count = {}
        tags_idx = {}
        for img_file in hf_fs.glob(f'{repo}/{step}/previews/*.png'):
            img_filename = os.path.basename(img_file)
            img_name = os.path.splitext(img_filename)[0]
            img_info_filename = f'{img_name}_info.txt'

            local_img_file = os.path.join(images_dir, img_filename)
            download_file(hf_hub_url(repo, filename=f'{step}/previews/{img_filename}'), local_img_file)
            local_info_file = os.path.join(images_dir, img_info_filename)
            download_file(hf_hub_url(repo, filename=f'{step}/previews/{img_info_filename}'), local_info_file)

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
            if not nsfw:
                cls_, score_ = nsfw_pred(local_img_file)
                if cls_ not in {'hentai', 'porn', 'sexy'} and score_ >= 0.65:
                    pass
                else:
                    nsfw = True

            if nsfw and not allow_nsfw_images:
                logging.info(f'Image {local_img_file!r} skipped due to its nsfw.')
                continue

            current_feat = ccip_extract_feature(local_img_file)
            similarity = ccip_batch_same([current_feat, *ccip_feats])[0, 1:].mean()
            logging.info(f'Similarity of character on image {local_img_file!r}: {similarity!r}')
            if similarity < 0.6 and not no_ccip_check:
                logging.info(f'Similarity of {local_img_file!r}({similarity!r}) is too low, skipped.')
                continue

            if not nsfw or allow_nsfw_images:
                rating_score = anime_rating_score(local_img_file)
                safe_v = int(round(rating_score['safe'] * 10))
                safe_r15 = int(round(rating_score['r15'] * 10))
                safe_r18 = int(round(rating_score['r18'] * 10))
                faces = detect_faces(local_img_file)
                if faces:
                    if len(faces) > 1:
                        logging.warning('Multiple face detected, skipped!')
                        continue

                    (x0, y0, x1, y1), _, _ = faces[0]
                    width, height = load_image(local_img_file).size
                    face_area = abs((x1 - x0) * (y1 - y0))
                    face_ratio = face_area * 1.0 / (width * height)
                    face_ratio = int(round(face_ratio * 50))
                else:
                    logging.warning('No face detected, skipped!')
                    continue

                images.append((
                    (-safe_v, -safe_r15, -safe_r18) if False else 0,
                    -face_ratio,
                    1 if nsfw else 0,
                    0 if img_name.startswith('pattern_') else 1,
                    img_name,
                    (local_img_file, img_filename, meta)
                ))

                for ptag in info.get('Prompt').split(','):
                    ptag = ptag.strip()
                    tags_count[ptag] = tags_count.get(ptag, 0) + 1
                    if ptag not in tags_idx:
                        tags_idx[ptag] = len(tags_idx)

        images = [item[-1] for item in sorted(images)]
        max_tag_cnt = max(tags_count.values())
        recommended_tags = sorted([ptag for ptag, cnt in tags_count.items() if cnt == max_tag_cnt],
                                  key=lambda x: tags_idx[x])

        # publish model
        session = session or get_civitai_session(timeout=30)

        model_desc_default = f"""
        * Thanks to Civitai's TOS, some images cannot be uploaded. **THE FULL PREVIEW IMAGES CAN BE FOUND ON [HUGGINGFACE](https://huggingface.co/{repo})**.
        * **<span style="color:#fa5252">THIS MODEL HAS TWO FILES. YOU NEED TO USE THEM TOGETHER!!!</span>**
        * **The associated trigger words are only for reference, it may need to be adjusted at some times**.
        * Recommended weight of pt file is 0.5-1.0, weight of LoRA is 0.5-0.85. 
        * Images were generated using a few fixed prompts and dataset-based clustered prompts. Random seeds were used, ruling out cherry-picking. **What you see here is what you can get.**
        * No specialized training was done for outfits. You can check our provided preview post to get the prompts corresponding to the outfits.
        * This model is trained with **{plural_word(dataset_size, "image")}**.

        ## How to Use This Model

        **<span style="color:#fa5252">THIS MODEL HAS TWO FILES. YOU NEED TO USE THEM TOGETHER!!!</span>**. 
        In this case, you need to download both `{pt_file}` and 
        `{lora_file}`, then **use `{pt_file}` as texture inversion embedding, and use
        `{lora_file}` as LoRA at the same time**.

        **<span style="color:#fa5252">このモデルには2つのファイルがあります。一緒に使う必要があります！！！</span>**。
        この場合、`{pt_file}`と`{lora_file}`の両方をダウンロード
        する必要があります。`{pt_file}`をテクスチャ反転埋め込みとして使用し、同時に`{lora_file}`をLoRAとして使用してください。

        **<span style="color:#fa5252">这个模型有两个文件。你需要同时使用它们！！！</span>**。
        在这种情况下，您需要下载`{pt_file}`和`{lora_file}`这两个文件，然后将`{pt_file}`用作纹理反转嵌入，
        同时使用`{lora_file}`作为LoRA。

        **<span style="color:#fa5252">이 모델은 두 개의 파일이 있습니다. 두 파일을 함께 사용해야 합니다!!!</span>**. 
        이 경우에는 `{pt_file}`와 `{lora_file}` 두 파일을 모두 다운로드하신 다음에 **`{pt_file}`을 텍스처 반전 임베딩으로 사용하고, 
        동시에 `{lora_file}`을 LoRA로 사용하셔야 합니다**.

        (Translated with ChatGPT)

        The trigger word is `{trigger_word}`, and the recommended tags are `{', '.join(recommended_tags)}`.

        ## How This Model Is Trained

        This model is trained with [HCP-Diffusion](https://github.com/7eu7d7/HCP-Diffusion). 
        And the auto-training framework is maintained by [DeepGHS Team](https://huggingface.co/deepghs).

        ## Why Some Preview Images Not Look Like {" ".join(map(str.capitalize, trigger_word.split("_")))}

        **All the prompt texts** used on the preview images (which can be viewed by clicking on the images) 
        **are automatically generated using clustering algorithms** based on feature information extracted from the 
        training dataset. The seed used during image generation is also randomly generated, and the images have 
        not undergone any selection or modification. As a result, there is a possibility of the mentioned 
        issues occurring.

        In practice, based on our internal testing, most models that experience such issues perform better in 
        actual usage than what is seen in the preview images. **The only thing you may need to do is adjusting 
        the tags you are using**.

        ## I Felt This Model May Be Overfitting or Underfitting, What Shall I Do

        Our model has been published on [huggingface repository - {repo}](https://huggingface.co/{repo}), where
        models of all the steps are saved. Also, we published the training dataset on 
        [huggingface dataset - {repo}](https://huggingface.co/datasets/{repo}), which may be helpful to you.

        ## Why Not Just Using The Better-Selected Images

        Our model's entire process, from data crawling, training, to generating preview images and publishing, 
        is **100% automated without any human intervention**. It's an interesting experiment conducted by our team, 
        and for this purpose, we have developed a complete set of software infrastructure, including data filtering, 
        automatic training, and automated publishing. Therefore, if possible, we would appreciate more feedback or 
        suggestions as they are highly valuable to us.

        ## Why Can't the Desired Character Outfits Be Accurately Generated

        Our current training data is sourced from various image websites, and for a fully automated pipeline, 
        it's challenging to accurately predict which official images a character possesses. 
        Consequently, outfit generation relies on clustering based on labels from the training dataset 
        in an attempt to achieve the best possible recreation. We will continue to address this issue and attempt 
        optimization, but it remains a challenge that cannot be completely resolved. The accuracy of outfit 
        recreation is also unlikely to match the level achieved by manually trained models.

        In fact, this model's greatest strengths lie in recreating the inherent characteristics of the characters 
        themselves and its relatively strong generalization capabilities, owing to its larger dataset. 
        As such, **this model is well-suited for tasks such as changing outfits, posing characters, and, 
        of course, generating NSFW images of characters!**😉".
        
        For the following groups, it is not recommended to use this model and we express regret:

        1. Individuals who cannot tolerate any deviations from the original character design, even in the slightest detail.
        2. Individuals who are facing the application scenarios with high demands for accuracy in recreating character outfits.
        3. Individuals who cannot accept the potential randomness in AI-generated images based on the Stable Diffusion algorithm.
        4. Individuals who are not comfortable with the fully automated process of training character models using LoRA, or those who believe that training character models must be done purely through manual operations to avoid disrespecting the characters.
        5. Individuals who finds the generated image content offensive to their values.
        """
        model_name = model_name or try_find_title(char_name, game_name) or \
                     try_get_title_from_repo(repo) or trigger_word.replace('_', ' ')
        if not force_create_model:
            try:
                exist_model = civitai_find_online(model_name, creator='narugo1992')
            except ModelNotFound:
                model_id = None
            else:
                logging.info(f'Existing model {exist_model.model_name}({exist_model.model_id}) found.')
                model_id = exist_model.model_id
        else:
            model_id = None

        model_id, _ = civitai_upsert_model(
            name=model_name,
            description_md=model_desc_md or model_desc_default,
            tags=[
                game_name, f"{game_name} {char_name}", char_name,
                'female', 'girl', 'character', 'fully-automated',
                *map(_tag_decode, core_tags.keys()),
            ],
            exist_model_id=model_id,
            session=session,
        )

        version_data = civitai_create_version(
            model_id=model_id,
            version_name=version_name,
            description_md=version_desc_md or '',
            trigger_words=[
                trigger_word,
                repr_tags([key for key, _ in sorted(core_tags.items(), key=lambda x: -x[1])]),
            ],
            session=session,
            steps=step,
            epochs=epoch,
        )
        version_id = version_data['id']

        civitai_upload_models(
            model_version_id=version_id,
            model_files=models,
            model_id=model_id,
            session=session,
        )
        civitai_upload_images(
            model_version_id=version_id,
            image_files=images,
            tags=[
                game_name, f"{game_name} {char_name}", char_name,
                'female', 'girl', 'character', 'fully-automated', 'random prompt', 'random seed',
                *map(_tag_decode, core_tags.keys()),
            ],
            model_id=model_id,
            session=session,
        )

        if draft:
            logging.info(f'Draft of model {model_id!r} created.')
        else:
            civiti_publish(model_id, version_id, publish_at, session)
        return civitai_get_model_info(model_id, session)['id']


def get_draft_models(session=None):
    session = session or get_civitai_session()
    resp = srequest(
        session, 'GET', 'https://civitai.com/api/trpc/model.getMyDraftModels',
        params={
            'input': json.dumps({"json": {"page": 1, "limit": 200, "authed": True}}),
        },
        headers={'Referer': f'https://civitai.com/user'},
    )
    return resp.json()['result']['data']['json']['items']


def delete_model(model_id: int, session=None):
    session = session or get_civitai_session()
    resp = srequest(
        session, 'POST', 'https://civitai.com/api/trpc/model.delete',
        json={"json": {"id": model_id, "permanently": False, "authed": True}},
        headers={'Referer': f'https://civitai.com/models/{model_id}'},
    )
    resp.raise_for_status()
