import glob
import json
import os.path
import re
import time
import zipfile
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
from natsort import natsorted
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

    if hf_fs.exists(f'datasets/{repository}/table.csv'):
        table_data = pd.read_csv(hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='table.csv',
        )).to_dict('records')
    else:
        table_data = []

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
            DanbooruSource([tag, 'order:random'])[:600].attach(
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
            ch_data = {
                'id': danbooru_id,
                'tag': tag,
                'short_tag': short_tag,
                'hprefix': hprefix,
                'post_count': post_count,
                'core_tags': core_tags,
            }
            all_characters.append(ch_data)
            with open(os.path.join(ch_dir, 'data.json'), 'w') as f:
                json.dump(ch_data, f, ensure_ascii=False, sort_keys=True, indent=4)
            with zipfile.ZipFile(os.path.join(ch_dir, 'images.zip'), 'w') as zf:
                for file in os.listdir(export_dir):
                    zf.write(os.path.join(export_dir, file), file)

            logging.info(f'Extracting feature of {tag!r} ...')
            feats = []
            for file in tqdm(glob.glob(os.path.join(export_dir, '*.webp'))):
                feats.append(ccip_extract_feature(file))
            np.save(os.path.join(ch_dir, 'feat.npy'), np.stack(feats))

            danbooru_wiki_url = f"https://danbooru.donmai.us/wiki_pages/{quote_plus(tag)}"
            sample_files = [
                os.path.relpath(file, upload_dir) for file in
                natsorted(glob.glob(os.path.join(ch_dir, '*.webp')))
            ]
            table_data.append({
                'Character': f'[{tag}]({danbooru_wiki_url})',
                'Sample1': f'![sample1]({sample_files[0]})',
                'Sample2': f'![sample2]({sample_files[1]})',
                'Sample3': f'![sample3]({sample_files[2]})',
                'Post Count': post_count,
                'Core Tags': f'`{", ".join(core_tags)}`',
            })

            with open(os.path.join(upload_dir, 'exist_tags.json'), 'w') as f:
                json.dump(sorted(exist_tags), f)
            with open(os.path.join(upload_dir, 'characters.json'), 'w') as f:
                json.dump(all_characters, f, ensure_ascii=False, indent=4, sort_keys=True)
            df = pd.DataFrame(table_data)
            df.to_csv(os.path.join(upload_dir, 'table.csv'), index=False)

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

                print('## Index', file=f)
                print('', file=f)

                print(df.to_markdown(index=False), file=f)
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
