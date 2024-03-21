import glob
import os.path
from typing import List, Tuple

import pandas as pd
from imgutils.sd import read_metadata


def find_steps_in_workdir_dep(workdir: str) -> Tuple[str, List[int]]:
    ckpts_dir = os.path.join(workdir, 'ckpts')
    pt_steps = []
    pt_name = None
    for pt in glob.glob(os.path.join(ckpts_dir, '*-*.pt')):
        name = os.path.basename(pt)
        segs = os.path.splitext(name)[0].split('-')
        if pt_name is None:
            pt_name = '-'.join(segs[:-1])
        else:
            if pt_name != '-'.join(segs[:-1]):
                raise NameError(f'Name not match, {pt_name!r} vs {"-".join(segs[:-1])!r}.')
        pt_steps.append(int(segs[-1]))

    unet_steps = []
    for unet in glob.glob(os.path.join(ckpts_dir, 'unet-*.safetensors')):
        name = os.path.basename(unet)
        segs = os.path.splitext(name)[0].split('-')
        unet_steps.append(int(segs[-1]))

    text_encoder_steps = []
    for text_encoder in glob.glob(os.path.join(ckpts_dir, 'text_encoder-*.safetensors')):
        name = os.path.basename(text_encoder)
        segs = os.path.splitext(name)[0].split('-')
        text_encoder_steps.append(int(segs[-1]))

    return pt_name, sorted(set(pt_steps) & set(unet_steps) & set(text_encoder_steps))


def find_steps_in_workdir(workdir: str) -> pd.DataFrame:
    data = []
    for file in glob.glob(os.path.join(workdir, 'kohya', '*.safetensors')):
        meta = read_metadata(file)
        step = int(meta['ss_steps'])
        data.append({
            'step': step,
            'epoch': int(meta['ss_epoch']),
            'name': os.path.splitext(os.path.basename(file))[0],
            'filename': os.path.basename(file),
            'file': os.path.abspath(file),
        })
    df = pd.DataFrame(data)
    df = df.sort_values(['step'], ascending=[True])
    return df
