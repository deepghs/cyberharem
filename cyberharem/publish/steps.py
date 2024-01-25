import glob
import os.path
from typing import List, Tuple


def find_steps_in_workdir(workdir: str) -> Tuple[str, List[int]]:
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
