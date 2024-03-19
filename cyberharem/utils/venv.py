import os
import shutil

from .path import get_path_from_env


def get_exec_from_venv(venv_dir: str, exec_name: str = 'python') -> str:
    venv_dir = os.path.abspath(venv_dir)
    paths_str = os.pathsep.join([
        os.path.join(venv_dir, 'bin'),
        os.path.join(venv_dir, 'Scripts'),
        get_path_from_env() or ''
    ])
    return shutil.which(exec_name, path=paths_str)


def get_python_exec_from_venv(venv_dir: str) -> str:
    return get_exec_from_venv(venv_dir, 'python')
