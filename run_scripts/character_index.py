import glob
import json
import os.path
import re
import time
import zipfile
from functools import lru_cache
from typing import List, Dict
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from ditk import logging
from hbutils.encoding import sha1
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory
from huggingface_hub import hf_hub_download
from imgutils.metrics import ccip_extract_feature
from imgutils.validate import anime_completeness
from tqdm import tqdm
from waifuc.action import NoMonochromeAction, FilterSimilarAction, \
    TaggingAction, PersonSplitAction, FaceCountAction, CCIPAction, ModeConvertAction, ClassFilterAction, \
    AlignMinSizeAction, FileExtAction, FileOrderAction, PaddingAlignAction, ArrivalAction, RatingFilterAction, \
    FilterAction
from waifuc.export import SaveExporter
from waifuc.model import ImageItem
from waifuc.source import DanbooruSource, LocalSource

from cyberharem.dataset.analysis import get_character_tags_info
from cyberharem.utils import get_hf_fs, get_hf_client


def _get_df_tags():
    df = pd.read_csv(hf_hub_download(
        repo_id='deepghs/site_tags',
        repo_type='dataset',
        filename='danbooru.donmai.us/tags.csv',
    ))
    df = df[df['category'] == 4]
    df = df.sort_values(['post_count'], ascending=False)
    df = df[df['post_count'] >= 500]
    return df


class CCFilterAction(FilterAction):
    def check(self, item: ImageItem) -> bool:
        type_, score = anime_completeness(item.image)
        return type_ == 'polished'


def _name_safe(name_text):
    return re.sub(r'[\W_]+', '_', name_text).strip('_')


@lru_cache()
def _db_session():
    s = DanbooruSource(['1girl'])
    s._prune_session()
    return s.session


def get_copyrights(tag, threshold: float = 0.7) -> List[str]:
    session = _db_session()
    resp = session.get('https://danbooru.donmai.us/related_tag.json', params={
        'query': tag,
        'category': 'copyright',
    })
    resp.raise_for_status()
    tags = []
    for item in resp.json()['related_tags']:
        if item['frequency'] >= threshold:
            tags.append((item['tag']['name'], item['frequency']))
    tags = sorted(tags, key=lambda x: (-x[1], x[0]))
    return [tag for tag, _ in tags]


def get_gender(tag) -> Dict[str, float]:
    session = _db_session()
    resp = session.get('https://danbooru.donmai.us/related_tag.json', params={
        'query': ' '.join([tag, 'solo']),
        'category': 'general',
    })
    resp.raise_for_status()

    boy_frequency, girl_frequency = None, None
    for item in resp.json()['related_tags']:
        if item['tag']['name'] == '1boy':
            boy_frequency = item['frequency']
        if item['tag']['name'] == '1girl':
            girl_frequency = item['frequency']
        if boy_frequency is not None and girl_frequency is not None:
            break

    boy_frequency = boy_frequency or 0.0
    girl_frequency = girl_frequency or 0.0
    logging.info(f'Gender frequency of {tag!r}, boy: {boy_frequency!r}, girl: {girl_frequency!r}')
    return {
        'boy': boy_frequency,
        'girl': girl_frequency,
    }


def run_it(repository: str, max_cnt: int, max_time_limit: float = 340 * 60, crawl_img_count: int = 50):
    start_time = time.time()

    hf_fs = get_hf_fs()
    hf_client = get_hf_client()

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset')
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    if hf_fs.exists(f'datasets/{repository}/exist_tags.json'):
        exist_tags = json.loads(hf_fs.read_text(f'datasets/{repository}/exist_tags.json'))
    else:
        exist_tags = []
    exist_tags = set(exist_tags)

    if hf_fs.exists(f'datasets/{repository}/characters.json'):
        all_characters = json.loads(hf_fs.read_text(f'datasets/{repository}/characters.json'))
    else:
        all_characters = []

    cnt = 0
    for item in _get_df_tags().to_dict('records'):
        if time.time() - start_time > max_time_limit:
            logging.info('Time reached, quit.')
            break
        if cnt >= max_cnt:
            logging.info('Max count reached, quit.')
            break

        danbooru_id = item['id']
        tag = item['name']
        short_tag = _name_safe(tag)
        post_count = item['post_count']
        hprefix = sha1(tag.encode())[:2]
        if tag in exist_tags:
            logging.info(f'Tag {tag!r} already crawled, skipped.')
            continue

        with TemporaryDirectory() as td:
            export_dir = os.path.join(td, 'export')
            os.makedirs(export_dir, exist_ok=True)

            logging.info('Crawling images from danbooru ...')
            DanbooruSource([tag, 'order:random'])[:800].attach(
                # preprocess images with white background RGB
                ModeConvertAction('RGB', 'white'),

                # pre-filtering for images
                CCFilterAction(),
                RatingFilterAction(['safe', 'r15']),
                NoMonochromeAction(),  # no monochrome, greyscale or sketch
                ClassFilterAction(['illustration', 'bangumi']),  # no comic or 3d
                # RatingFilterAction(['safe', 'r15']),  # filter images with rating, like safe, r15, r18
                FilterSimilarAction('all'),  # filter duplicated images

                # human processing
                FaceCountAction(1),  # drop images with 0 or >1 faces
                PersonSplitAction(),  # crop for each person
                FaceCountAction(1),

                ArrivalAction('Before CCIP'),
                # CCIP, filter the character you may not want to see in dataset
                CCIPAction(min_val_count=15),

                # if min(height, weight) > 800, resize it to 800
                AlignMinSizeAction(640),

                # tagging with wd14 v2, if you don't need character tag, set character_threshold=1.01
                TaggingAction(force=True),

                FilterSimilarAction('all'),  # filter again
                # MirrorAction(),  # mirror image for data augmentation
                FileExtAction(ext='.webp'),  # random rename files
            )[:crawl_img_count].export(SaveExporter(export_dir))

            img_count = len(glob.glob(os.path.join(export_dir, '*.webp')))
            if img_count < 10:
                exist_tags.add(tag)
                logging.warning(f'Too few valid images detect for {tag!r}, skipped.')
                continue
            core_tags, _ = get_character_tags_info(LocalSource(export_dir))

            upload_dir = os.path.join(td, 'upload')
            os.makedirs(upload_dir, exist_ok=True)

            ch_dir = os.path.join(upload_dir, hprefix, short_tag)
            os.makedirs(ch_dir)
            logging.info('Generating samples ...')
            ss1 = LocalSource(export_dir, shuffle=True).attach(RatingFilterAction(['safe']))
            ss2 = LocalSource(export_dir, shuffle=True).attach(RatingFilterAction(['r15']))
            (ss1 + ss2).attach(
                PaddingAlignAction((512, 768)),
                FileOrderAction(ext='.webp'),
            )[:3].export(ch_dir)

            exist_tags.add(tag)
            copyrights = get_copyrights(tag)
            copyright = copyrights[0] if copyrights else None
            ch_data = {
                'id': danbooru_id,
                'tag': tag,
                'short_tag': short_tag,
                'hprefix': hprefix,
                'post_count': post_count,
                'core_tags': core_tags,
                'copyright': copyright,
                'copyrights': copyrights,
                'gender': get_gender(tag),
            }
            all_characters.append(ch_data)
            with open(os.path.join(ch_dir, 'data.json'), 'w') as f:
                json.dump(ch_data, f, ensure_ascii=False, sort_keys=True, indent=4)
            with zipfile.ZipFile(os.path.join(ch_dir, 'images.zip'), 'w') as zf:
                for file in os.listdir(export_dir):
                    zf.write(os.path.join(export_dir, file), file)

            pages_dir = os.path.join(upload_dir, 'pages')
            os.makedirs(pages_dir, exist_ok=True)
            pages_map = {}
            for ch_item in all_characters:
                if ch_item['copyright'] not in pages_map:
                    pages_map[ch_item['copyright']] = []

                def _rel_path(ft):
                    return os.path.relpath(
                        os.path.join(upload_dir, ch_item['hprefix'], ch_item['short_tag'], ft),
                        pages_dir
                    )

                danbooru_wiki_url = f'https://danbooru.donmai.us/wiki_pages/{quote_plus(ch_item["tag"])}'
                pages_map[ch_item['copyright']].append({
                    'Character': f'[{ch_item["tag"]}]({danbooru_wiki_url})',
                    'Sample1': f'![sample1]({_rel_path("1.webp")})',
                    'Sample2': f'![sample2]({_rel_path("2.webp")})',
                    'Sample3': f'![sample3]({_rel_path("3.webp")})',
                    'Post Count': ch_item['post_count'],
                    'Core Tags': f'`{", ".join(ch_item["core_tags"])}`',
                })

            cp_data = []
            for ch_copyright, ch_items in sorted(pages_map.items(),
                                                 key=lambda x: (0 if x[0] else 1, -len(x[1]), x[0])):
                ch_md_file = os.path.join(pages_dir, f'{_name_safe(ch_copyright or "unknown")}.md')
                with open(ch_md_file, 'w') as f:
                    print(f'# {ch_copyright}', file=f)
                    print(f'', file=f)

                    if ch_copyright:
                        copyright_wiki_url = f'https://danbooru.donmai.us/wiki_pages/{quote_plus(ch_copyright)}'
                        print(f'Danbooru tag: `{ch_copyright}`, wiki page: {copyright_wiki_url}', file=f)
                        print(f'', file=f)

                    print(f'This dataset if for collecting all the hot characters from the internet, '
                          f'and extract their features and core tags. '
                          f'It will be useful for **automatically testing the character generating ability of '
                          f'the anime-style base models**.', file=f)
                    print(f'', file=f)
                    print(f'{plural_word(len(ch_items), "character")} in total.', file=f)
                    print('', file=f)

                    print('## Character Index', file=f)
                    print('', file=f)
                    print(pd.DataFrame(ch_items).to_markdown(index=False), file=f)
                    print('', file=f)

                copyright_page_relpath = os.path.relpath(ch_md_file, upload_dir)
                cp_data.append({
                    'Copyright': f'[{ch_copyright or "(unknown)"}]({copyright_page_relpath})',
                    'Count': len(pages_map[ch_copyright]),
                })

            logging.info(f'Extracting feature of {tag!r} ...')
            feats = []
            for file in tqdm(glob.glob(os.path.join(export_dir, '*.webp'))):
                feats.append(ccip_extract_feature(file))
            np.save(os.path.join(ch_dir, 'feat.npy'), np.stack(feats))

            with open(os.path.join(upload_dir, 'exist_tags.json'), 'w') as f:
                json.dump(sorted(exist_tags), f)
            with open(os.path.join(upload_dir, 'characters.json'), 'w') as f:
                json.dump(all_characters, f, ensure_ascii=False, indent=4, sort_keys=True)

            df_cp = pd.DataFrame(cp_data)
            with open(os.path.join(upload_dir, 'README.md'), 'w') as f:
                print('---', file=f)
                print('license: mit', file=f)
                print('---', file=f)
                print('', file=f)

                print('# Anime Character Index', file=f)
                print('', file=f)

                print(f'This dataset if for collecting all the hot characters from the internet, '
                      f'and extract their features and core tags. '
                      f'It will be useful for **automatically testing the character generating ability of '
                      f'the anime-style base models**.', file=f)
                print(f'', file=f)
                print(f'{plural_word(len(all_characters), "character")} in total.', file=f)
                print('', file=f)

                print('## Copyrights', file=f)
                print('', file=f)
                print(df_cp.to_markdown(index=False), file=f)
                print('', file=f)

            upload_directory_as_directory(
                repo_id=repository,
                repo_type='dataset',
                local_directory=upload_dir,
                path_in_repo='.',
                message=f'Upload {tag}\'s information'
            )
            cnt += 1


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    run_it(
        repository='deepghs/character_index',
        max_cnt=200000,
        max_time_limit=340 * 60,
    )
