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

## Install

Clone and install this project

```shell
git clone https://github.com/deepghs/cyberharem.git
cd cyberharem
pip install -r requirements.txt
```

This project works on HuggingFace. You should **set the namespace on HuggingFace before start using it**

```shell
export CH_NAMESPACE=my_hf_username
export HF_TOKEN=your_huggingface_token
```

After set `CH_NAMESPACE`, your datasets or models will be saved to `my_hf_username/xxxxx`.

## Dataset Making

### Create Dataset With Waifuc

Here is the [cyberharem project](https://github.com/deepghs/cyberharem), an efficient train data collector for anime
waifu.
We recommend you to learn how to use it before start reading this
part: https://deepghs.github.io/cyberharem/main/index.html

After that, run the following code

```python
from cyberharem.source import DanbooruSource

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
from cyberharem.source import LocalSource

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

## Train P-LoRA

The training method we employ is pivotal tuning, which stores the trigger words of LoRA in an embedding file. The
activation of LoRA is achieved by triggering the embedding file during use. We refer to this as P-LoRA.

Before we start, we should init the configuration files of HCP framework

```shell
hcpinit

```

You can train a P-LoRA with the dataset on huggingface

```python
from cyberharem.train import train_plora

# workdir is the directory to save the trained loras
workdir = train_plora(
    ds_repo_id='CyberHarem/surtr_arknights',

    # how many ckeckpoints you want to keep
    keep_ckpts=40,

    # select a based model (this one is NAI)
    pretrained_model='deepghs/animefull-latest',

    bs=4,  # batch size
    max_reg_bs=16,  # batch size of regularization dataset
    train_resolution=720,  # resolution to train

    # total steps to train will be auto calculated based on bs and your dataset's size
    # approx: min(max(dataset_size * max_epochs / bs, min_steps), max(dataset_size * min_epochs / bs, max_steps))
    min_epochs=10,
    max_epochs=40,
    min_steps=800,
    max_steps=10000,

    pt_lr=0.03,  # learning rate of embedding
    unet_lr=1e-4,  # learning rate of unet (text encoder will not be trained)
    unet_rank=0.01,  # rank of unet
)
```

Please note that this script takes about 28G GPU memory in maximum. We can run it on A100 80G, but maybe you cannot
run it on 2060. If OOM occurred, just lower the `bs` and `max_reg_bs`.

## Evaluate LoRA and Publish It To HuggingFace

```python
from cyberharem.publish import deploy_to_huggingface

deploy_to_huggingface(
    workdir='runs/surtr_arknights',
    eval_cfgs=dict(
        # batch size to infer
        # this number is for A100 80G
        # please lower it if you do not have so much GPU memory
        batch_size=32,

        # model to create images
        pretrained_model='Meina/MeinaMix_V11',
        model_hash=None,  # fill this hash, or the images will not be referenced to the base model on civitai

        # arguments for sd inference
        firstpass_width=512,
        firstpass_height=768,
        width=832,
        height=1216,
        cfg_scale=7,
        infer_steps=30,
        sample_method='DPM++ 2M Karras',
        lora_alpha=0.8,
    )
)
```

Images will be created for steps evaluation. After that, best steps will be recommended, and all the information
(images, model files, data archives and LoRAs) will be pushed to model repository `my_hf_username/surtr_arknights`.

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

