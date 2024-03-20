import mimetypes
import os
from typing import Iterator

mimetypes.add_type('image/webp', '.webp')


def is_image_file(file):
    gtype = mimetypes.guess_type(os.path.normcase(file))
    if gtype:
        gtype = gtype[0]
    else:
        gtype = None
    return 'image' in (gtype or '')


def is_npz_file(file):
    _, ext = os.path.splitext(os.path.normcase(file))
    return ext == '.npz'


def is_txt_file(file):
    _, ext = os.path.splitext(os.path.normcase(file))
    return ext == '.txt'


def yield_all_images(dataset_dir) -> Iterator[str]:
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            src_file = os.path.join(root, file)
            if is_image_file(src_file):
                yield src_file
