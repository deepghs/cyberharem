import json
from typing import Mapping, Optional, List


def _yaml_recursive(data, segments: Optional[list] = None):
    segments = list(segments or [])
    if isinstance(data, Mapping):
        for key, value in data.items():
            yield from _yaml_recursive(value, [*segments, key])
    else:
        key = '.'.join(map(str, segments))
        value = json.dumps(data)
        yield f'{key}={value}'


def data_to_cli_args(data) -> List[str]:
    return list(_yaml_recursive(data))
