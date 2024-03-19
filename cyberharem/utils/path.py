import os


def get_path_from_env():
    path = os.environ.get("PATH", None)
    if path is None:
        try:
            path = os.confstr("CS_PATH")
        except (AttributeError, ValueError):
            # os.confstr() or CS_PATH is not available
            path = os.defpath

    return path
