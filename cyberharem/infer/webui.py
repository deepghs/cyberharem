import glob
import logging
import os.path
from typing import Optional

from hbutils.random import random_sha1_with_timestamp
from webuiapi import WebUIApi

_WEBUI_CLIENT: Optional[WebUIApi] = None


def set_webui_server(host="127.0.0.1", port=7860, baseurl=None, use_https=False, **kwargs):
    global _WEBUI_CLIENT
    logging.info(f'Set webui server {"https" if use_https else "http"}://{host}:{port}/{baseurl or ""}')
    _WEBUI_CLIENT = WebUIApi(
        host=host,
        port=port,
        baseurl=baseurl,
        use_https=use_https,
        **kwargs
    )


def _get_webui_client() -> WebUIApi:
    if _WEBUI_CLIENT:
        return _WEBUI_CLIENT
    else:
        raise OSError('Webui server not set, please set that with `set_webui_server` function.')


class LoraMock:
    def mock_lora(self, local_lora_file: str) -> str:
        raise NotImplementedError

    def unmock_lora(self, lora_name: str):
        raise NotImplementedError


class LocalLoraMock(LoraMock):
    def __init__(self, sd_webui_dir: str):
        self.sd_webui_dir = sd_webui_dir
        self.lora_dir = os.path.abspath(os.path.join(self.sd_webui_dir, 'models', 'Lora', 'automation'))
        os.makedirs(self.lora_dir, exist_ok=True)

    def mock_lora(self, local_lora_file: str) -> str:
        random_sha = random_sha1_with_timestamp()
        _, ext = os.path.splitext(local_lora_file)
        src_file = os.path.abspath(local_lora_file)
        dst_file = os.path.join(self.lora_dir, f'{random_sha}{ext}')
        logging.info(f'Mocking lora file from {src_file!r} to {dst_file!r} ...')
        os.symlink(src_file, dst_file)
        return random_sha

    def unmock_lora(self, lora_name: str):
        files = glob.glob(os.path.join(self.lora_dir, f'{lora_name}.*'))
        if files:
            file = files[0]
            if os.path.islink(file):
                logging.info(f'Unmocking lora {lora_name!r} from {file!r} ...')
                os.unlink(file)
            else:
                raise RuntimeError(f'Mocked lora file {file!r} is not a sym link, cannot unmock.')
        else:
            raise RuntimeError(f'No mocked lora file {lora_name!r} found.')


_WEBUI_LORA_MOCK: Optional[LoraMock] = None


def set_webui_local_dir(webui_local_dir: str):
    global _WEBUI_LORA_MOCK
    logging.info(f'Setting webui local directory {webui_local_dir!r} ...')
    _WEBUI_LORA_MOCK = LocalLoraMock(webui_local_dir)


def _get_webui_lora_mock() -> LoraMock:
    return _WEBUI_LORA_MOCK
