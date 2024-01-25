import os
from typing import Optional, Literal, Dict, Tuple, Union, List, Mapping, Any

from .session import get_requests_session, srequest

ActionStatusTyping = Literal[
    'completed', 'action_required', 'cancelled', 'failure', 'neutral', 'skipped', 'stale', 'success', 'timed_out',
    'in_progress', 'queued', 'requested', 'waiting', 'pending'
]


class GithubActionClient:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get('GH_TOKEN')
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._session = get_requests_session(headers=headers)

        self._workflow_ids: Dict[Tuple[str, str], int] = {}

    def list_runs(self, repo_id: str, workflow_id: Union[str, int],
                  status: Optional[ActionStatusTyping] = None, page: int = 1, per_page: int = 100):
        if isinstance(workflow_id, str):
            workflow_id = self._get_workflow_id(repo_id, workflow_id)

        params = {
            'page': str(page),
            'per_page': str(per_page),
        }
        if status:
            params['status'] = status
        resp = srequest(
            self._session, 'GET',
            f'https://api.github.com/repos/{repo_id}/actions/workflows/{workflow_id}/runs',
            params=params
        )
        return resp.json()

    def list_all_runs(self, repo_id: str, workflow_id: Union[str, int], status: Optional[ActionStatusTyping] = None):
        page = 1
        retval = []
        while True:
            data = self.list_runs(repo_id, workflow_id, status, page)
            if not data['workflow_runs']:
                break

            retval.extend(data['workflow_runs'])
            page += 1

        return retval

    def list_all_unfinished_runs(self, repo_id: str, workflow_id: Union[str, int]):
        unfinished_status: List[ActionStatusTyping] = ['in_progress', 'queued', 'requested', 'waiting', 'pending']
        retval = []
        for status in unfinished_status:
            retval.extend(self.list_all_runs(repo_id, workflow_id, status))

        return retval

    def get_run(self, repo_id: str, run_id: int):
        resp = srequest(
            self._session, 'GET',
            f'https://api.github.com/repos/{repo_id}/actions/runs/{run_id}',

        )
        return resp.json()

    def list_workflows(self, repo_id: str):
        resp = srequest(
            self._session, 'GET',
            f'https://api.github.com/repos/{repo_id}/actions/workflows',
        )
        return resp.json()

    def _get_workflow_id(self, repo_id: str, name: str):
        if (repo_id, name) not in self._workflow_ids:
            for item in self.list_workflows(repo_id)['workflows']:
                if item['name'] == name:
                    self._workflow_ids[(repo_id, name)] = item['id']
                    break
            else:
                raise ValueError(f'Workflow {name!r} not found for repository {repo_id!r}.')

        return self._workflow_ids[(repo_id, name)]

    def create_workflow_run(self, repo_id: str, workflow_id: Union[str, int], data: Mapping[str, Any]):
        if isinstance(workflow_id, str):
            workflow_id = self._get_workflow_id(repo_id, workflow_id)
        srequest(
            self._session, 'POST',
            f'https://api.github.com/repos/{repo_id}/actions/workflows/{workflow_id}/dispatches',
            json={
                'ref': 'main',
                'inputs': data,
            }
        )
