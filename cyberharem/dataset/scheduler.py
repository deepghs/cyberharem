import json
import os
import warnings
from typing import Literal, Union, List

from ditk import logging
from gchar.games import get_character_class
from gchar.games.base import Character
from gchar.generic import import_generic
from gchar.resources.pixiv import get_pixiv_posts
from hbutils.string import plural_word
from tqdm.auto import tqdm

from .crawler import DATASET_PVERSION
from ..utils import get_ch_name, get_hf_client, get_hf_fs, get_global_namespace
from ..utils.ghaction import GithubActionClient

logging.try_init_root(logging.INFO)
logger = logging.getLogger("pyrate_limiter")
logger.disabled = True

import_generic()


class Task:
    def __init__(self, ch):
        self.ch = ch

    @property
    def repo_id(self):
        return f'{get_global_namespace()}/{get_ch_name(self.ch)}'

    @property
    def character_name(self):
        return str(self.ch.enname or self.ch.jpname or self.ch.cnname or '')

    @property
    def game_name(self):
        return self.ch.__class__.__game_name__

    def __eq__(self, other):
        return type(self) == type(other) and self.ch == other.ch

    def __hash__(self):
        return hash((self.ch,))

    def __repr__(self):
        return f'<Task character: {self.ch!r}, game: {self.ch.__class__.__game_name__!r}>'


def _get_pixiv_posts(ch_: Character):
    ret = get_pixiv_posts(ch_)
    if ret:
        return ret[0]
    else:
        return 0


def _get_character_list_from_game_cls(game_cls):
    return [
        ch
        for ch in sorted(game_cls.all(), key=lambda x: (-_get_pixiv_posts(x), x))
        if ch.gender == 'female' and not ch.is_extra
    ]


TaskStatusTyping = Literal['not_started', 'on_going', 'completed']


class Scheduler:
    def __init__(self, game_name: Union[str, List[str]], concurrent: int = 6, max_new_create: int = 3):
        self.game_clses = [
            get_character_class(game_name)
            for game_name in (game_name if isinstance(game_name, list) else [game_name])
        ]
        self.concurrent = concurrent
        self.max_new_create = max_new_create

    def list_task_pool(self):
        all_girls = []
        for game_cls in self.game_clses:
            all_girls.extend(_get_character_list_from_game_cls(game_cls))

        tasks, repo_id_set = [], set()
        for ch in all_girls:
            task = Task(ch)
            try:
                _repo_id = task.repo_id
            except (ValueError,) as err:
                warnings.warn(f'Error: {err!r} for task {task!r}, skipped.')
                continue

            if _repo_id in repo_id_set:
                continue

            tasks.append(task)
            repo_id_set.add(_repo_id)

        return tasks

    def get_task_status(self, task: Task) -> TaskStatusTyping:
        hf_client = get_hf_client()
        hf_fs = get_hf_fs()

        if not hf_client.repo_exists(repo_id=task.repo_id, repo_type='dataset'):
            return 'not_started'
        if hf_fs.exists(f'datasets/{task.repo_id}/.git-empty'):
            return 'completed'
        if hf_fs.exists(f'datasets/{task.repo_id}/.git-ongoing'):
            return 'on_going'

        if not hf_fs.exists(f'datasets/{task.repo_id}/README.md'):
            return 'not_started'
        md_text = hf_fs.read_text(f'datasets/{task.repo_id}/README.md')
        if 'outfit' not in md_text.lower():
            return 'not_started'

        if not hf_fs.exists(f'datasets/{task.repo_id}/meta.json'):
            return 'not_started'
        meta_text = hf_fs.read_text(f'datasets/{task.repo_id}/meta.json')
        if 'Waifuc-Raw' not in meta_text:
            return 'not_started'
        meta_info = json.loads(meta_text)
        version = meta_info.get('version')
        if version == DATASET_PVERSION:
            return 'completed'
        else:
            return 'not_started'

    def go_up(self):
        client = GithubActionClient()

        on_goings = []
        not_started = []
        completed = []
        for task in tqdm(self.list_task_pool()):
            try:
                status = self.get_task_status(task)
            except (ValueError,) as err:
                warnings.warn(f'Error: {err!r} for task {task!r}, skipped.')
                continue

            logging.info(f'Task {task!r}, status: {status!r}')
            if status == 'on_going':
                on_goings.append(task)
            elif status == 'not_started':
                not_started.append(task)
            elif status == 'completed':
                completed.append(task)
            else:
                assert False, 'Should not reach this line.'

        logging.info(f'{plural_word(len(completed), "completed task")}, '
                     f'{plural_word(len(on_goings), "on-going task")}, '
                     f'{plural_word(len(not_started), "not started task")}.')

        x = len(on_goings)
        hf_client = get_hf_client()
        i = 0
        new_create_cnt = 0
        while x < self.concurrent and i < len(not_started):
            task: Task = not_started[i]
            _repo_exists = hf_client.repo_exists(repo_id=task.repo_id, repo_type='dataset')
            if task.character_name and (_repo_exists or new_create_cnt < self.max_new_create):
                logging.info(f'Scheduling for {task!r} ...')
                client.create_workflow_run(
                    'deepghs/cyberharem',
                    'Test Script',
                    data={
                        'character_name': task.character_name,
                        'game_name': task.game_name,
                        'drop_multi': False,
                    }
                )

                x += 1
                if not _repo_exists:
                    new_create_cnt += 1

            i += 1


_DEFAULT_CONCURRENCY = 12
_MAX_NEW_CREATE = 3

_ALL_GAMES = [
    # 'bangdream',
    # 'bangdreamdai2ki',
    #
    # 'theidolmster',
    # 'theidolmstermillionlive',
    # 'theidolmstershinycolors',
    # 'idolmastercinderellagirls',

    # 'lovelive',
    # 'lovelivenijigasakihighschoolidolclub',
    # 'loveliveschoolidolfestivalallstars',
    # 'lovelivesunshine',
    # 'lovelivesuperstar',
    #
    # 'honkai3',
    # 'fireemblem',

    'neuralcloud',
]

if __name__ == '__main__':
    concurrency = int(os.environ.get('CH_CONCURRENCY') or _DEFAULT_CONCURRENCY)
    logging.info(f'Concurrency: {concurrency!r}')
    s = Scheduler(_ALL_GAMES, concurrent=concurrency, max_new_create=_MAX_NEW_CREATE)
    s.go_up()
