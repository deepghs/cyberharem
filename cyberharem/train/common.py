def train_common_lora(repository):
    from .train import train_lora
    from ..publish import deploy_to_huggingface
    workdir = train_lora(repository)
    deploy_to_huggingface(workdir, repository=repository)


def train_tiny_lora(repository, resolution: int = 512, epochs: int = 15):
    from .train import train_lora
    from ..publish import deploy_to_huggingface
    workdir = train_lora(
        repository,
        use_reg=True,
        eps=epochs,
        save_interval=1,
        resolution=resolution,
    )
    deploy_to_huggingface(
        workdir,
        repository=repository,
        ccip_distance_mode=True,
        ccip_check=None,
        eval_cfgs=dict(
            fidelity_alpha=1.45,
        ),
    )
