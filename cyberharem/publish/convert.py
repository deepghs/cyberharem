import logging
import os
from typing import Union, List, Mapping

import safetensors
import torch
from hcpdiff.ckpt_manager import auto_manager
from hcpdiff.tools.lora_convert import LoraConverter
from safetensors.torch import save_file as safetensors_save_file


def convert_to_webui_lora(lora_path, lora_path_te, dump_path, auto_scale_alpha: bool = True):
    converter = LoraConverter()

    # load lora model
    if lora_path_te:
        logging.info(f'Converting lora model {lora_path!r} and {lora_path_te!r} to {dump_path!r} ...')
    else:
        logging.info(f'Converting lora model {lora_path!r} (unet only) to {dump_path!r} ...')
    ckpt_manager = auto_manager(lora_path)

    sd_unet = ckpt_manager.load_ckpt(lora_path)
    if lora_path_te:
        sd_te = ckpt_manager.load_ckpt(lora_path_te)
    else:
        sd_te = None
    state = converter.convert_to_webui(
        sd_unet=sd_unet['lora'],
        sd_TE=sd_te['lora'] if sd_te is not None else None,
        auto_scale_alpha=auto_scale_alpha,
        sdxl=False,
    )
    logging.info(f'Saving Webui LoRA to {dump_path!r} ...')
    # noinspection PyProtectedMember
    ckpt_manager._save_ckpt(state, save_path=dump_path)


def load_state_dict(file_path):
    is_safetensors = os.path.splitext(os.path.basename(file_path))[-1] == '.safetensors'
    if is_safetensors:
        state_dict = {}
        with safetensors.safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        state_dict = torch.load(file_path)
    return state_dict


def pack_to_bundle_lora(lora_model: str, embeddings: Union[List[str], Mapping[str, str]],
                        bundle_lora_path: str):
    lora_sd = load_state_dict(lora_model)
    if isinstance(embeddings, list):
        embs_sd = {
            os.path.splitext(os.path.basename(x))[0]: load_state_dict(x)
            for x in embeddings
        }
    else:
        embs_sd = {name: load_state_dict(x) for name, x in embeddings.items()}

    for emb, emb_sd in embs_sd.items():
        for key, value in emb_sd.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    lora_sd[f"bundle_emb.{emb}.{key}.{subkey}"] = subvalue
            elif isinstance(value, torch.Tensor):
                lora_sd[f"bundle_emb.{emb}.{key}"] = value

    for k in lora_sd:
        if k.startswith("bundle_emb"):
            logging.info(f'Exising bundle emb keys in lora {lora_model!r}: {k!r}')

    is_safetensors = os.path.splitext(os.path.basename(bundle_lora_path))[-1] == ".safetensors"
    if is_safetensors:
        logging.info(f'Saving bundled lora to {bundle_lora_path!r} with safetensors format.')
        safetensors_save_file(lora_sd, bundle_lora_path)
    else:
        logging.info(f'Saving bundled lora to {bundle_lora_path!r} with torch format.')
        torch.save(lora_sd, bundle_lora_path)
