import logging
import re
import time
import warnings

from pyanimeinfo.myanimelist import JikanV4Client
from requests import HTTPError
from thefuzz import fuzz

jikan_client = JikanV4Client()


def _name_safe(name_text):
    return re.sub(r'[\W_]+', ' ', name_text).strip(' ')


def search_from_myanimelist(title: str):
    logging.info('Search information from myanimelist ...')
    while True:
        try:
            jikan_items = jikan_client.search_anime(query=title)
        except HTTPError as err:
            if err.response.status_code == 429:
                warnings.warn(f'429 error detected: {err!r}, wait for some seconds ...')
                time.sleep(5.0)
            else:
                raise
        else:
            break
    collected_aitems = []
    type_map = {'tv': 0, 'movie': 1, 'ova': 2, 'ona': 2}
    for i, pyaitem in enumerate(jikan_items):
        if not pyaitem['type'] or pyaitem['type'].lower() not in type_map:
            continue

        max_partial_ratio = max(
            fuzz.partial_ratio(
                _name_safe(title).lower(),
                _name_safe(title_item['title']).lower(),
            ) for title_item in pyaitem['titles']
        ) / 100.0
        max_ratio = max(
            fuzz.ratio(
                _name_safe(title).lower(),
                _name_safe(title_item['title']).lower(),
            ) for title_item in pyaitem['titles']
        ) / 100.0
        if max_partial_ratio > 0.9:
            collected_aitems.append((
                -max_partial_ratio, -max_ratio, type_map[pyaitem['type'].lower()], i, pyaitem))

    if not collected_aitems:
        logging.warning('No information found on myanime list, skipped.')
        return None
    else:
        collected_aitems = sorted(collected_aitems)
        _, _, _, _, myanime_item = collected_aitems[0]
        logging.info(f'Found on myanimelist: {myanime_item["url"]!r} ...')
        return myanime_item
