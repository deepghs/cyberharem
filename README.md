# CyberHarem

![GitHub Org's stars](https://img.shields.io/github/stars/deepghs)
[![GitHub stars](https://img.shields.io/github/stars/deepghs/cyberharem)](https://github.com/deepghs/cyberharem/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/deepghs/cyberharem)](https://github.com/deepghs/cyberharem/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/deepghs/cyberharem)
[![GitHub issues](https://img.shields.io/github/issues/deepghs/cyberharem)](https://github.com/deepghs/cyberharem/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/deepghs/cyberharem)](https://github.com/deepghs/cyberharem/pulls)
[![Contributors](https://img.shields.io/github/contributors/deepghs/cyberharem)](https://github.com/deepghs/cyberharem/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/deepghs/cyberharem)](https://github.com/deepghs/cyberharem/blob/master/LICENSE)

CyberHarem Automated Waifu Training Pipeline

(NOTE: This project is still work in progress. It has only been tested on A100 80G, ubuntu environment.)

**(NOTE: HCP-Diffusion has been kicked out from CyberHarem. Now CyberHarem is based on kohya script and a41 webui.)**

## Install

Clone and install this project

```shell
git clone https://github.com/deepghs/cyberharem.git
cd cyberharem
pip install -r requirements.txt
```

This project works on HuggingFace. You should **set the namespace on HuggingFace before start using it**

```shell
# set your huggingface username or organization name
export CH_NAMESPACE=my_hf_username

# set your huggingface token
export HF_TOKEN=your_huggingface_token
```

After set `CH_NAMESPACE`, your datasets or models will be saved to `my_hf_username/xxxxx`.

## Dataset Making

### Create Dataset With Waifuc

Here is the [waifuc project](https://github.com/deepghs/waifuc), an efficient train data collector for anime
waifu.
We recommend you to learn how to use it before start reading this
part: https://deepghs.github.io/waifuc/main/index.html

After that, run the following code

```python
from waifuc.source import DanbooruSource

from cyberharem.dataset import crawl_dataset_to_huggingface

s = DanbooruSource(['surtr_(arknights)'])

crawl_dataset_to_huggingface(
    # your cyberharem datasource
    source=s,

    # name of dataset, trigger word of model
    name='surtr_arknights',

    # display name (for others to see, e.g. on civitai)
    display_name='surtr/スルト/史尔特尔 (Arknights)',

    # how many images you need,
    limit=500,
)

```

The dataset with 500 original images will be pushed to dataset repository `my_hf_username/surtr_arknights`.
This step may take several hours.

It is worth noting that you do not have to add attached actions after the source. They will be added inside the
`crawl_dataset_to_huggingface` function, and the datasource will be auto-cleaned and processed.

In some cases, if you do not want it to process your dataset (e.g. your datasource is trusted or processed), just run
as following

```python
from waifuc.source import LocalSource

from cyberharem.dataset import crawl_dataset_to_huggingface

s = LocalSource('/my/local/directory')

crawl_dataset_to_huggingface(
    source=s,
    name='surtr_arknights',
    display_name='surtr/スルト/史尔特尔 (Arknights)',

    # no limit on the quantity
    limit=None,

    # skip all the pre-processes 
    skip_preprocess=True,
)
```

### Batch Process Anime Videos

First, you need to download the anime videos to your local environment (e.g. at folder `/my/anime/videos`)

```shell
set CH_BG_NAMESPACE=bg_namespace
python -m cyberharem.dataset.video huggingface -i /my/anime/videos -n 'Name of The Anime'
```

Then the bangumi dataset will be pushed to `bg_namespace/nameoftheanime`.

More options can be found with `-h` option

```
Usage: python -m cyberharem.dataset.video huggingface [OPTIONS]

  Publish to huggingface

Options:
  -r, --repository TEXT   Repository to publish to.
  -R, --revision TEXT     Revision for pushing the model.  [default: main]
  -i, --input TEXT        Input videos.  [required]
  -n, --name TEXT         Bangumi name  [required]
  -s, --min_size INTEGER  Min size of image.  [default: 320]
  -E, --no_extract        No extraction from videos.
  -h, --help              Show this message and exit.
```

### Extract Training Dataset from Bangumi Dataset

You can extract images from the bangumi dataset (e.g. `BangumiBase/fatestaynightufotable`,
abovementioned `bg_namespace/nameoftheanime`), like this

```python
from cyberharem.dataset import crawl_base_to_huggingface

crawl_base_to_huggingface(
    # bangumi repository id
    source_repository='BangumiBase/fatestaynightufotable',

    ch_id=[18, 19],  # index numbers in bangumi repository
    name='Illyasviel Von Einzbern',  # official name of this waifu
    limit=1000,  # max number of images you need
)
```

Then the bangumi-based dataset will be uploaded to `my_hf_username/illyasviel_von_einzbern_fatestaynightufotable`.
Like this: https://huggingface.co/datasets/CyberHarem/illyasviel_von_einzbern_fatestaynightufotable .

## Train LoRA

~~The training method we employ is pivotal tuning, which stores the trigger words of LoRA in an embedding file. The
activation of LoRA is achieved by triggering the embedding file during use. We refer to this as P-LoRA.~~

That is all history, now we use kohya script to train common LoRAs.

You can train a LoRA with the dataset on huggingface

(PS: if you need to use reg dataset for training, please set the `REG_HOME` directory, this directory is used for reg
dataset and latent cache management.)

```python
from ditk import logging

from cyberharem.train import train_lora, set_kohya_from_conda_dir, set_kohya_from_venv_dir

logging.try_init_root(logging.INFO)

# if your kohya script is in conda
set_kohya_from_conda_dir(
    # name of the conda environment
    conda_env_name='kohya',

    # directory of kohya sd-script
    kohya_directory='/my/path/sd-script',
)

# # if your kohya script is in venv
# set_kohya_from_venv_dir(
#     # these should be a venv folder in this directory
#     kohya_directory='/my/path/sd-script',
#    
#     # name of venv, default is `venv`
#     venv_name='venv',
# )

if __name__ == '__main__':
    workdir = train_lora(
        ds_repo_id='CyberHarem/surtr_arknights',

        # use your own template file
        # this one is the default config template, you can just use it
        template_file='ch_lora_sd15.toml',

        # use reg dataset
        use_reg=True,

        # hyperparameters for training
        bs=8,  # training batch size
        unet_lr=0.0006,  # learning date of unet
        te_lr=0.0006,  # learning rate of text encoder
        train_te=False,  # do not train text encoder
        dim=4,  # dim of lora
        alpha=2,  # alpha of lora
        resolution=720,  # resolution: 720x720
        res_ratio=2.2,  # min_res: 720 // 2.2, max_res: 720 * 2.2
    )

```

Please note that this script takes about 18G GPU memory in maximum. We can run it on A100 80G, but maybe you cannot
run it on 2060. If OOM occurred, just lower the `bs` and `max_reg_bs`.

Also, you can specify directory or environment information of kohya script, like the followings:

* Set kohya in conda environment

```shell
export CH_KOHYA_DIR=/my/path/sd-script
export CH_KOHYA_CONDA_ENV=kohya
unset CH_KOHYA_VENV
```

* Set kohya in venv

```shell
export CH_KOHYA_DIR=/my/path/sd-script
export CH_KOHYA_VENV=venv
unset CH_KOHYA_CONDA_ENV
```

By using these variables, you do NOT have to specify them in your python code.

## Evaluate LoRA and Publish It To HuggingFace

We can automatically use a1111's webui to generate images, assess which LoRA step is the best, and publish them to the
huggingface hub.

NOTE:

1. **[Dynamic Prompts Plugin](https://github.com/adieyal/sd-dynamic-prompts) is REQUIRED for image batch
   generation!!!** Please install it before batch inference.
2. Please start your webui with API mode, by using `--api` and `--nowebui` arguments.

```python
from cyberharem.infer import set_webui_server, set_webui_local_dir
from cyberharem.publish import deploy_to_huggingface

# your a41 webui server
set_webui_server('127.0.0.1', 10188)

# your directory of a41 webui
# these should have `models/Lora` inside
set_webui_local_dir('/my/a41_webui/stable-diffusion-webui')

deploy_to_huggingface(
    workdir='runs/surtr_arknights',  # work directory of training
    eval_cfgs=dict(
        # basic infer arguments
        base_model='meinamix_v11',  # use the base model in a41 webui
        batch_size=64,  # we can use bs64 on A100 80G, lower this value if you cant
        sampler_name='DPM++ 2M Karras',
        cfg_scale=7,
        steps=30,
        firstphase_width=512,
        firstphase_height=768,
        clip_skip=2,

        # hires fix
        enable_hr=True,
        hr_resize_x=832,
        hr_resize_y=1216,
        denoising_strength=0.6,
        hr_second_pass_steps=20,
        hr_upscaler='R-ESRGAN 4x+ Anime6B',

        # adetailer, useful for fixing the eyes
        # will be ignored when adetailer not installed
        enable_adetailer=True,

        # weight of lora
        lora_alpha=0.8,
    )
)

```

Images will be created for steps evaluation. After that, best steps will be recommended, and all the information
(images, model files, data archives and LoRAs) will be pushed to model repository `my_hf_username/surtr_arknights`.

Also, if you do not want to set webui settings in python code, just use the following environment variables

```shell
export CH_WEBUI_SERVER=http://127.0.0.1:10188
export CH_WEBUI_DIR=/my/a41_webui/stable-diffusion-webui
```

## Upload to CivitAI

Before uploading, you need to create a civitai session
with [civitai_client](https://github.com/narugo1992/civitai_client).

```python
from cyberharem.publish import civitai_upload_from_hf

civitai_upload_from_hf(
    repository='my_hf_username/surtr_arknights',
    civitai_session='your_civitai_session.json',

    # use best step, you can use the step you like best
    step=None,

    # upload nsfw images (please attention the TOS of civitai)
    allow_nsfw=True,

    publish_at=None,  # publish now
    # publish_at='2030-01-01 08:00:00+00:00', # schedule to publish at '2030-01-01 08:00:00+00:00'

    # if you have already uploaded an older version, put the model id here
    # existing_model_id=None,
)
```

## F.A.Q.

### Will Private Repository Or Local Directory Be Supported?

No, and never will be. We developed and open-sourced this project with the intention of making the training of waifus
simpler and more convenient, while also ensuring more stable quality control. Resources such as datasets and models
should belong to all anime waifu enthusiasts. They are created by a wide range of anime artists and are collected and
compiled fully automatically by tools like cyberharem and cyberharem. **Our hope is for these resources to be widely
circulated, rather than monopolized in any form**. If you do not agree with this philosophy, we do not recommend that
you continue using this project.

