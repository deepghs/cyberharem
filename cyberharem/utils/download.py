import os
from contextlib import contextmanager

import requests
from tqdm.auto import tqdm

from .session import get_requests_session, srequest


class _FakeClass:
    def update(self, *args, **kwargs):
        pass


@contextmanager
def _with_tqdm(expected_size, desc, silent: bool = False):
    """
    Context manager that provides a tqdm progress bar for tracking the download progress.

    :param expected_size: The expected size of the file being downloaded.
    :type expected_size: int
    :param desc: The description of the progress bar.
    :type desc: str
    :param silent: Whether to silence the progress bar. If True, a fake progress bar is used. (default: False)
    :type silent: bool
    """
    if not silent:
        with tqdm(total=expected_size, unit='B', unit_scale=True, unit_divisor=1024, desc=desc) as pbar:
            yield pbar
    else:
        yield _FakeClass()


def download_file(url, filename, expected_size: int = None, desc=None, session=None, silent: bool = False, **kwargs):
    """
    Downloads a file from the given URL and saves it to the specified filename.

    :param url: The URL of the file to download.
    :type url: str
    :param filename: The filename to save the downloaded file to.
    :type filename: str
    :param expected_size: The expected size of the file in bytes. (default: None)
    :type expected_size: int
    :param desc: The description of the download progress. If not provided, the filename is used. (default: None)
    :type desc: str
    :param session: An existing requests Session object to use for the download. If not provided, a new Session object is created. (default: None)
    :type session: requests.Session
    :param silent: Whether to silence the progress bar. If True, no progress bar is displayed. (default: False)
    :type silent: bool
    :param kwargs: Additional keyword arguments to pass to the `srequest` function.
    :type kwargs: dict
    :returns: The filename of the downloaded file.
    :rtype: str
    """
    session = session or get_requests_session()
    response = srequest(session, 'GET', url, stream=True, allow_redirects=True, **kwargs)
    expected_size = expected_size or response.headers.get('Content-Length', None)
    expected_size = int(expected_size) if expected_size is not None else expected_size

    desc = desc or os.path.basename(filename)
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(filename, 'wb') as f:
        with _with_tqdm(expected_size, desc, silent) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                pbar.update(len(chunk))

    actual_size = os.path.getsize(filename)
    if expected_size is not None and actual_size != expected_size:
        os.remove(filename)
        raise requests.exceptions.HTTPError(f"Downloaded file is not of expected size, "
                                            f"{expected_size} expected but {actual_size} found.")

    return filename
