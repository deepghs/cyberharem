import logging

from hcpdiff.ckpt_manager import auto_manager
from hcpdiff.tools.lora_convert import LoraConverter


def convert_to_webui_lora(lora_path, lora_path_TE, dump_path, auto_scale_alpha: bool = True):
    converter = LoraConverter()

    # load lora model
    logging.info(f'Converting lora model {lora_path!r} and {lora_path_TE!r} to {dump_path!r} ...')
    ckpt_manager = auto_manager(lora_path)()

    sd_unet = ckpt_manager.load_ckpt(lora_path)
    sd_TE = ckpt_manager.load_ckpt(lora_path_TE)
    state = converter.convert_to_webui(sd_unet['lora'], sd_TE['lora'])
    if auto_scale_alpha:
        state = {k: (v * v.shape[1] if 'lora_up' in k else v) for k, v in state.items()}
    ckpt_manager._save_ckpt(state, save_path=dump_path)
