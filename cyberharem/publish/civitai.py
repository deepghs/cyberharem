import json
import logging
import os.path
import re
from typing import Optional

import markdown_strings
import numpy as np
import pandas as pd
from civitai.client import CivitAIClient
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import download_archive_as_directory, download_file_to_file
from huggingface_hub import hf_hub_url
from imgutils.data import load_image
from imgutils.detect import detect_faces
from imgutils.sd import get_sdmeta_from_image
from imgutils.tagging import remove_underline
from imgutils.validate import anime_rating
from pycivitai import civitai_find_online
from pycivitai.client import ModelNotFound, ModelVersionNotFound
from tqdm import tqdm

from cyberharem.utils import get_hf_fs


def _detect_face_value(image_file) -> float:
    img = load_image(image_file)
    detects = detect_faces(img)
    if detects:
        (x0, y0, x1, y1), _, _ = detects[0]
        return abs((x1 - x0) * (y1 - y0)) / (img.width * img.height)
    else:
        return 0.0


def _extract_words_from_display_name(display_name: str):
    return [word.strip() for word in re.findall(r'[\w \'_]+', display_name)]


def civitai_upload_from_hf(repository: str, step: Optional[int] = None, allow_nsfw: bool = False,
                           civitai_session: Optional[str] = None, publish_at: Optional[str] = None,
                           draft: bool = False, update_description_only: bool = False,
                           existing_model_id: Optional[int] = None, version_name: Optional[str] = None):
    hf_fs = get_hf_fs()
    meta_info = json.loads(hf_fs.read_text(f'{repository}/meta.json'))
    step = step or meta_info['best_step']
    if step not in meta_info['steps']:
        raise ValueError(f'Unknown step, one of {meta_info["steps"]!r} expected, but {step!r} given.')

    with TemporaryDirectory() as td:
        metrics_file = os.path.join(td, 'metrics.csv')
        download_file_to_file(repo_id=repository, repo_type='model',
                              file_in_repo='metrics.csv', local_file=metrics_file)
        df_metrics = pd.read_csv(metrics_file)

    step_info = df_metrics[df_metrics['step'] == step].to_dict('records')[0]
    name = meta_info['name']
    dataset_size = meta_info['dataset']['size']
    epoch = step_info['epoch']

    with TemporaryDirectory() as td:
        download_archive_as_directory(
            local_directory=td,
            file_in_repo=f'{step}/{name}.zip',
            repo_id=repository,
            repo_type='model',
            hf_token=os.environ.get('HF_TOKEN'),
        )

        local_metrics_plot_png = os.path.join(td, f'metrics_plot.png')
        download_file_to_file(
            local_file=local_metrics_plot_png,
            file_in_repo=f'metrics_plot.png',
            repo_id=repository,
            repo_type='model',
            hf_token=os.environ.get('HF_TOKEN'),
        )

        lora_file = f'{name}.safetensors'
        lora_path = os.path.join(td, lora_file)
        pt_file = f'{name}.pt'

        details_csv_file = os.path.join(td, 'details.csv')
        download_file_to_file(
            local_file=details_csv_file,
            file_in_repo=f'{step}/details.csv',
            repo_id=repository,
            repo_type='model',
            hf_token=os.environ.get('HF_TOKEN'),
        )

        metrics_info = json.loads(hf_fs.read_text(f'{repository}/{step}/metrics.json'))
        df = pd.read_csv(details_csv_file)
        df['level'] = [
            2 if 'smile' in img_file or 'portrait' in img_file else (
                1 if 'pattern' in img_file else 0
            )
            for img_file in tqdm(df['image'], desc='Calculating face area')
        ]
        df['face'] = [
            _detect_face_value(os.path.join(td, img_file))
            for img_file in tqdm(df['image'], desc='Calculating face area')
        ]
        df['rating'] = [
            anime_rating(os.path.join(td, img_file))[0]
            for img_file in tqdm(df['image'], desc='Detecting rating')
        ]

        df = df[df['ccip'] >= (metrics_info['ccip'] - 0.05)]
        df = df[df['bp'] >= (metrics_info['bp'] - 0.05)]
        df = df[df['aic'] >= max(metrics_info['aic'] - 0.3, metrics_info['aic'] * 0.5)]

        df['ccip_x'] = np.round(df['ccip'] * 30) / 30.0
        df['face_x'] = np.round(df['face'] * 20) / 20.0
        df = df.sort_values(by=['ccip_x', 'level', 'face_x'], ascending=False)
        upload_groups = [
            ('main', False, df[df['rating'] == 'safe']),
            ('r15', False, df[df['rating'] == 'r15']),
        ]
        if not allow_nsfw:
            df = df[df['rating'] != 'r18']
        else:
            upload_groups.append(('r18', True, df[df['rating'] == 'r18']))

        images_to_upload = [os.path.join(td, img_item) for img_item in df['image']]
        logging.info(f'{plural_word(len(images_to_upload), "image")} to upload.')

        model_hashes = []
        for img_file in images_to_upload:
            sd_meta = get_sdmeta_from_image(img_file)
            if sd_meta.parameters.get('Model hash'):
                model_hashes.append(sd_meta.parameters['Model hash'])
        model_hashes = sorted(set(model_hashes))
        model_ids, version_ids = [], []
        for model_hash in model_hashes:
            try:
                model_info = civitai_find_online(model_hash)
            except ModelNotFound:
                logging.info(f'Model {model_hash!r} not found in civitai.')
                continue
            else:
                logging.info(f'Model {model_hash!r} found, '
                             f'model name: {model_info.model_name!r}, model id: {model_info.model_id!r}, '
                             f'version name: {model_info.version_name!r}, version id: {model_info.version_id!r}.')
                model_ids.append(model_info.model_id)
                version_ids.append(model_info.version_id)
        model_ids = sorted(set(model_ids))
        version_ids = sorted(set(version_ids))

        civitai_session = civitai_session or os.environ.get('CIVITAI_SESSION')
        logging.info(f'Checking civitai session based on {civitai_session!r} ...')
        client = CivitAIClient.load(civitai_session)
        whoami = client.whoami
        if whoami:
            logging.info(f'Session confirmed {whoami!r} ...')
        else:
            raise RuntimeError(f'Session not found in {civitai_session!r}.')

        dataset_info = meta_info['dataset']
        metrics_plot_url = client.upload_image(local_metrics_plot_png).original_url

        description_md = f"""
        * Due to Civitai's TOS, some images cannot be uploaded. **THE FULL PREVIEW IMAGES CAN BE FOUND ON [HUGGINGFACE](https://huggingface.co/{repository})**.
        * For models version v1.5.1 or v2.0+, **you can simply use them on webui like other LoRAs, they are trained on kohya script.**
        * For models version v1.5 or v1.4-, you have to use both 2 files to run it. see "How to use Pivotal Tuned models" in description for details.
        * **The pruned character tags are {markdown_strings.esc_format(", ".join(map(remove_underline, meta_info['core_tags'])))}. You can add them into prompts when core features (e.g. hair color) of character is not so stable**.
        * Recommended weight of pt file is 0.7-1.1, weight of LoRA is 0.5-0.85. 
        * Images were generated using some fixed prompts and dataset-based clustered prompts. Random seeds were used, ruling out cherry-picking. **What you see here is what you can get.**
        * No specialized training was done for outfits. You can check our provided preview post to get the prompts corresponding to the outfits.
        * This model is trained with **{plural_word(dataset_size, "image")}**.
        * Training configuration file is [here]({hf_hub_url(repo_id=repository, repo_type='model', filename=f'train.toml')}).
        * **The step we auto-selected is {step} to balance the fidelity and controllability of the model.**
        Here is the overview of all the steps. You can try the other recommended steps in
        [huggingface repository - {markdown_strings.esc_format(repository)}](https://huggingface.co/{repository}).

        ![Step Overview]({metrics_plot_url})

        ## How to Use This Model

        **This part is only for model version v1.5.1 or v2.0+.**

        You can simply use it like other LoRAs. We trained this model with kohya scripts.

        他のLoRAと同様に簡単に使用できます。このモデルはkohyaスクリプトで訓練されました。

        다른 LoRA처럼 간단히 사용할 수 있습니다. 우리는 이 모델을 kohya 스크립트로 훈련했습니다.

        您可以像其他LoRAs一样简单地使用它。我们使用kohya脚本对该模型进行了训练。

        (Translated with ChatGPT)

        If you are looking for model waifus and or interested in our technology, 
        you can enter our discord server: https://discord.gg/AhExhbzxq9

        ## How This Model Is Trained

        * This model is **trained with [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)**,
        the images are generated with [a1111\'s webui](AUTOMATIC1111/stable-diffusion-webui) 
        and [API sdk](https://github.com/mix1009/sdwebuiapi).
        * The [auto-training framework](https://github.com/deepghs/cyberharem) is maintained by 
        [DeepGHS Team](https://huggingface.co/deepghs).
        * Dataset used for training is the `{dataset_info["name"]}` in 
        [{markdown_strings.esc_format(dataset_info["repository"])}](https://huggingface.co/datasets/{dataset_info["repository"]}),
        which contains {plural_word(dataset_info["size"], "image")}.
        * **The step we auto-selected is {step} to balance the fidelity and controllability of the model.
        * Training configuration file is [here]({hf_hub_url(repo_id=repository, repo_type='model', filename=f'train.toml')}).

        For more training details and recommended steps, take a look at 
        [huggingface repository - {markdown_strings.esc_format(repository)}](https://huggingface.co/{repository}).

        ## How to Use Pivotal Tuned Models

        **This part is only for model version v1.5 or v1.4-.**

        **<span style="color:#fa5252">THIS MODEL HAS TWO FILES. YOU NEED TO USE THEM TOGETHER IF YOU ARE USING WEBUI v1.6 OR LOWER VERSION!!!</span>**. 
        In this case, you need to download both `{pt_file}` and `{lora_file}`, 
        then **put `{pt_file}` to the embeddings folder, and use `{lora_file}` as LoRA at the same time**.
        **If you are using webui v1.7+, just use the safetensors file like the common LoRAs.**
        This is because the embedding-bundled LoRA/Lycoris model are now officially supported by a1111's webui, 
        see [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/13568) for more details.

        **<span style="color:#fa5252">このモデルには2つのファイルがあります。WebUI v1.6 以下のバージョンを使用している場合は、これらを一緒に使用する必要があります！！！</span>**
        この場合、`{pt_file}` と `{lora_file}` の両方をダウンロードする必要があり、
        **その後、`{pt_file}` を `embeddings` フォルダに入れ、同時に `{lora_file}` をLoRAとして使用します**。
        **webui v1.7+を使用している場合、一般的なLoRAsのようにsafetensorsファイルを使用してください。**
        これは、埋め込みバンドルされたLoRA/Lycorisモデルが現在、a1111のwebuiに公式にサポートされているためです。
        詳細については[こちら](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/13568)をご覧ください。

        **<span style="color:#fa5252">此模型包含两个文件。如果您使用的是 WebUI v1.6 或更低版本，您需要同时使用这两个文件！</span>**
        在这种情况下，您需要下载 `{pt_file}` 和 `{lora_file}` 两个文件，
        然后**将 `{pt_file}` 放入 `embeddings` 文件夹中，并同时使用 `{lora_file}` 作为 LoRA**。
        **如果您正在使用 webui v1.7 或更高版本，只需像常规 LoRAs 一样使用 safetensors 文件。**
        这是因为嵌入式 LoRA/Lycoris 模型现在已经得到 a1111's webui 的官方支持，
        更多详情请参见[这里](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/13568)。

        (Translated with ChatGPT)

        The trigger word is `{name}`, and the pruned tags are `{', '.join(meta_info["core_tags"])}`.
        **When some features (e.g. hair color) are not so stable at some times, 
        you can add these them into your prompt**.

        ## Why Some Preview Images Not Look Like Her

        **All the prompt texts** used on the preview images (which can be viewed by clicking on the images) 
        **are automatically generated using clustering algorithms** based on feature information extracted from the 
        training dataset. The seed used during image generation is also randomly generated, and the images have 
        not undergone any selection or modification. As a result, there is a possibility of the mentioned 
        issues occurring.

        In practice, based on our internal testing, most models that experience such issues perform better in 
        actual usage than what is seen in the preview images. **The only thing you may need to do is adjusting 
        the tags you are using**.

        ## I Felt This Model May Be Overfitting or Underfitting, What Shall I Do

        **The step you see here is auto-selected**. We also recommend other good steps for you to try.
        Click [here](https://huggingface.co/{repository}#which-step-should-i-use) to select your favourite step.
        
        Our model has been published on 
        [huggingface repository - {markdown_strings.esc_format(repository)}](https://huggingface.co/{repository}), 
        where models of all the steps are saved. Also, we published the training dataset on 
        [huggingface dataset - {markdown_strings.esc_format(repository)}](https://huggingface.co/datasets/{repository}), 
        which may be helpful to you.

        ## Why Not Just Using The Better-Selected Images

        Our model's entire process, from data crawling, training, to generating preview images and publishing, 
        is **100% automated without any human intervention**. It's an interesting experiment conducted by our team, 
        and for this purpose, we have developed a complete set of software infrastructure, including data filtering, 
        automatic training, and automated publishing. Therefore, if possible, we would appreciate more feedback or 
        suggestions as they are highly valuable to us.

        ## Why Can't the Desired Character Outfits Be Accurately Generated

        Our current training data is sourced from various image websites, and for a fully automated pipeline, 
        it's challenging to accurately predict which official images a character possesses. 
        Consequently, outfit generation relies on clustering based on labels from the training dataset 
        in an attempt to achieve the best possible recreation. We will continue to address this issue and attempt 
        optimization, but it remains a challenge that cannot be completely resolved. The accuracy of outfit 
        recreation is also unlikely to match the level achieved by manually trained models.

        In fact, this model's greatest strengths lie in recreating the inherent characteristics of the characters 
        themselves and its relatively strong generalization capabilities, owing to its larger dataset. 
        As such, **this model is well-suited for tasks such as changing outfits, posing characters, and, 
        of course, generating NSFW images of characters!**😉".
        
        For the following groups, it is not recommended to use this model and we express regret:

        1. Individuals who cannot tolerate any deviations from the original character design, even in the slightest detail.
        2. Individuals who are facing the application scenarios with high demands for accuracy in recreating character outfits.
        3. Individuals who cannot accept the potential randomness in AI-generated images based on the Stable Diffusion algorithm.
        4. Individuals who are not comfortable with the fully automated process of training character models using LoRA, or those who believe that training character models must be done purely through manual operations to avoid disrespecting the characters.
        5. Individuals who finds the generated image content offensive to their values.
        """

        # create or update model
        tags = [*_extract_words_from_display_name(meta_info['display_name']), *meta_info['core_tags']]
        if not existing_model_id:
            try:
                existing_model_id = civitai_find_online(meta_info['display_name'], creator=client.whoami.name).model_id
            except ModelNotFound:
                existing_model_id = None

        model_info = client.upsert_model(
            name=meta_info['display_name'],
            description_md=description_md,
            tags=tags,
            category='character',
            type_='LORA',
            # checkpoint_type='Trained',  # use this line when uploading checkpoint
            commercial_use=['Image', 'RentCivit', 'Rent'],  # your allowance of commercial use
            allow_no_credit=True,
            allow_derivatives=True,
            allow_different_licence=True,
            nsfw=False,
            poi=False,
            exist_model_id=existing_model_id,
            # if your model has already been uploaded, put its id here to avoid duplicated creation
        )

        if update_description_only:
            logging.info('Only update description, quit.')
            return

        # create or update version
        version_name = version_name or meta_info['version']
        if existing_model_id:
            try:
                existing_version_id = civitai_find_online(existing_model_id, version_name,
                                                          creator=client.whoami.name).version_id
            except (ModelVersionNotFound,):
                existing_version_id = None
        else:
            existing_version_id = None
        version_info = client.upsert_version(
            model_id=model_info['id'],
            version_name=version_name,
            description_md=f'Model {markdown_strings.esc_format(name)} version {version_name}',
            trigger_words=[name],
            base_model='SD 1.5',
            epochs=epoch,
            steps=step,
            clip_skip=2,
            vae_name=None,
            early_access_time=0,
            recommended_resources=version_ids,  # put the version ids of the resources here, e.g. 119057 is meinamix v11
            require_auth_when_download=False,
            exist_version_id=existing_version_id,
            # if this version already exist, put its version id here to avoid duplicated creation
        )

        # upload model files
        client.upload_models(
            model_version_id=version_info['id'],
            model_files=[lora_path],
        )

        post_ids = []
        for group_name, group_is_nsfw, df_group in upload_groups:
            local_images = [os.path.join(td, img_item) for img_item in df_group['image']]
            if local_images:
                logging.info(f'{plural_word(len(local_images), "image")} to upload for group {group_name!r}.')
                post_id = client.upload_images_for_model_version(
                    model_version_id=version_info['id'],
                    image_files=local_images,
                    tags=[*tags, group_name],  # tags of images
                    nsfw=group_is_nsfw,
                )
                post_ids.append(post_id)
            else:
                logging.info(f'No images to upload for group {group_name!r}.')

        # publish the model
        if not draft:
            if not existing_model_id:
                client.model_publish(
                    model_id=model_info['id'],
                    model_version_id=version_info['id'],
                    publish_at=publish_at,  # publish it at once when None
                    # publish_at='10 days later',  # schedule the publishing time
                )
            else:
                client.model_version_publish(
                    model_version_id=version_info['id'],
                    publish_at=publish_at,  # publish it at once when None
                    # publish_at='10 days later',  # schedule the publishing time
                )

            for post_id in post_ids:
                client.post_publish(
                    post_id=post_id,
                    publish_at=publish_at,  # publish it at once when None
                    # publish_at='10 days later',  # schedule the publishing time
                )

        # set associated resources
        if model_ids:
            client.model_set_associate(
                model_id=model_info['id'],
                resource_ids=model_ids,  # suggested model ids, e.g. 7240 is meinamix
            )
