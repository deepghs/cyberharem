import glob
import json
import logging
import os.path
import pathlib
from typing import Optional, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from natsort import natsorted
from sdeval.controllability import BikiniPlusMetrics
from sdeval.corrupt import AICorruptMetrics
from sdeval.fidelity import CCIPMetrics
from tqdm import tqdm

from ..infer import draw_images_for_workdir
from ..infer.draw import _DEFAULT_INFER_MODEL, _DEFAULT_INFER_CFG_FILE_LORA


def list_steps(workdir) -> List[int]:
    ckpts_dir = os.path.join(workdir, 'ckpts')
    return sorted(set([
        int(os.path.splitext(os.path.basename(file))[0].split('-')[-1])
        for file in glob.glob(os.path.join(ckpts_dir, 'unet-*.safetensors'))
    ]))


def plt_metrics(df: pd.DataFrame, model_name: str, dataset_size: int, plot_file: str,
                bs: int = 4, select: int = 5, fidelity_alpha: float = 3.0,
                bp_std_min: Optional[float] = 0.015, min_aic: float = 0.8, aic_interval: float = 0.05,
                select_n: int = 3):
    # generate data calculation
    df = df.copy()
    df['epoch'] = np.ceil(df['step'] * bs / dataset_size).astype(np.int32)

    ccip = np.array(df['ccip'])
    ccip_mean = ccip.mean()
    ccip_std = ccip.std()
    ccip_norm = np.clip(((ccip - ccip_mean) / ccip_std) * 0.2 + 0.6, a_min=1e-6, a_max=1.0)
    df['ccip_norm'] = ccip_norm

    bp = np.array(df['bp'])
    bp_mean = bp.mean()
    bp_std = bp.std()
    if bp_std_min is not None:
        bp_std = np.maximum(bp_std, bp_std_min)
    bp_norm = np.clip(((bp - bp_mean) / bp_std) * 0.2 + 0.6, a_min=1e-6, a_max=1.0)
    df['bp_norm'] = bp_norm

    df['integrate'] = (fidelity_alpha ** 2 + 1) * (ccip_norm * bp_norm) / (fidelity_alpha ** 2 * bp_norm + ccip_norm)
    df = df[['step', 'epoch', 'ccip', 'aic', 'bp', 'ccip_norm', 'bp_norm', 'integrate']]

    # select best steps
    d = 1.0
    while True:
        if len(df[df['aic'] >= d]) >= select * select_n:
            break
        d -= aic_interval
        if d < min_aic:
            d = None
            break

    if d is None:
        if len(df[df['aic'] >= min_aic]) >= select * (select_n - 1):
            df_aic = df[df['aic'] >= min_aic]
        else:
            df_aic = df.sort_values(by=['aic'], ascending=False)[:select * (select_n - 1)]
    else:
        df_aic = df[df['aic'] >= d]

    aic_steps = df_aic['step'].tolist()
    aic_failed_steps = df[~df['step'].isin(aic_steps)]['step'].tolist()

    df_selected = df_aic.sort_values(by=['integrate', 'ccip', 'aic', 'bp'], ascending=False)[:select]
    selected_steps = df_selected['step'].tolist()
    aic_unselected_steps = sorted(set(aic_steps) - set(selected_steps))
    best_step = selected_steps[0]
    best_step_info = df_selected[df_selected['step'] == best_step].to_dict('records')[0]
    non_best_steps = sorted(set(selected_steps) - {best_step})

    # plot metrics
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(12, 9))

    axes[0, 0].plot(df['step'], df['ccip'])
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('CCIP Score')
    axes[0, 0].set_title('Fidelity')
    axes[0, 0].grid()

    axes[0, 1].plot(df['step'], df['bp'])
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Bp-Avg Score')
    axes[0, 1].set_title('Controllability')
    axes[0, 1].grid()

    axes[1, 0].plot(df['step'], df['aic'])
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('AI Corrupt Score')
    axes[1, 0].set_ylim([0.7, 1.0])
    axes[1, 0].axhspan(0.95, 1.0, facecolor='#dbf7c2')
    axes[1, 0].axhspan(0.9, 0.95, facecolor='#f4f7c2')
    axes[1, 0].axhspan(0.8, 0.9, facecolor='#f7e4c2')
    axes[1, 0].axhspan(0.7, 0.8, facecolor='#f7c7c2')
    axes[1, 0].set_title('Vision Corrupt')
    axes[1, 0].grid()

    axes[1, 1].plot(df['step'], df['integrate'])
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel(f'F-{fidelity_alpha:.1f} Score')
    axes[1, 1].set_title('Integration')
    axes[1, 1].grid()

    axes[1, 1].scatter(
        df[df['step'].isin(aic_failed_steps)]['step'],
        df[df['step'].isin(aic_failed_steps)]['integrate'],
        marker='x', c='red'
    )
    axes[1, 1].scatter(
        df[df['step'].isin(aic_unselected_steps)]['step'],
        df[df['step'].isin(aic_unselected_steps)]['integrate'],
        marker='o', c='orange'
    )
    axes[1, 1].scatter(
        df[df['step'].isin(non_best_steps)]['step'],
        df[df['step'].isin(non_best_steps)]['integrate'],
        marker='o', c='green'
    )
    axes[1, 1].scatter(
        df[df['step'] == best_step]['step'],
        df[df['step'] == best_step]['integrate'],
        marker='*', c='green', s=128
    )

    plt.suptitle(f'Model of {model_name!r} (alpha={fidelity_alpha})\n'
                 f'Selected Steps: {selected_steps!r}\n'
                 f'Best Step: {best_step}, CCIP: {best_step_info["ccip"]:.3f}, '
                 f'B-P: {best_step_info["bp"]:.3f}, AI-C: {best_step_info["aic"]:.3f}')
    plt.savefig(plot_file, dpi=150)
    plt.cla()

    return df, df_selected


def eval_for_workdir(workdir: str, batch_size: int = 32,
                     pretrained_model: str = _DEFAULT_INFER_MODEL, model_hash: Optional[str] = None,
                     firstpass_width: int = 512, firstpass_height: int = 768,
                     width: int = 832, height: int = 1216, cfg_scale: float = 7,
                     infer_steps: int = 30, hires_steps: int = 18,
                     lora_alpha: float = 0.8, clip_skip: int = 2, sample_method: str = 'DPM++ 2M Karras',
                     select: int = 5, fidelity_alpha: float = 3.0,
                     cfg_file: str = _DEFAULT_INFER_CFG_FILE_LORA, model_tag: str = 'lora'):
    logging.info(f'Evaluate for workdir {workdir!r} ...')
    with open(os.path.join(workdir, 'meta.json'), 'r') as f:
        meta_info = json.load(f)

    steps = list_steps(workdir)
    logging.info(f'Steps {steps!r} found.')

    features_path = os.path.join(workdir, 'features.npy')
    logging.info(f'Loading features from {features_path!r} ...')
    ccip_metrics = CCIPMetrics(None, feats=np.load(features_path))
    bp_metrics = BikiniPlusMetrics()
    aic_metrics = AICorruptMetrics()

    eval_dir = os.path.join(workdir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    tb_data = []
    for step in tqdm(steps):
        step_dir = os.path.join(eval_dir, str(step))
        os.makedirs(step_dir, exist_ok=True)
        step_metrics_file = os.path.join(step_dir, 'metrics.json')
        if not os.path.exists(step_metrics_file):
            logging.info(f'Drawing images with step {step!r} in workdir {workdir!r} ...')
            drawings = draw_images_for_workdir(
                workdir=workdir,
                model_steps=step,
                batch_size=batch_size,
                pretrained_model=pretrained_model,
                model_hash=model_hash,
                firstpass_width=firstpass_width,
                firstpass_height=firstpass_height,
                width=width,
                height=height,
                cfg_scale=cfg_scale,
                infer_steps=infer_steps,
                hires_steps=hires_steps,
                model_alpha=lora_alpha,
                clip_skip=clip_skip,
                sample_method=sample_method,
                cfg_file=cfg_file,
                model_tag=model_tag,
            )
            logging.info(f'Saving images to {step_dir!r} ...')
            for drawing in tqdm(drawings):
                drawing.save(os.path.join(step_dir, f'{drawing.name}.png'))

            pathlib.Path(step_metrics_file).touch()

        step_details_file = os.path.join(step_dir, 'details.csv')
        if not os.path.exists(step_details_file):
            png_files = natsorted(glob.glob(os.path.join(step_dir, '*.png')))
            png_filenames = [os.path.relpath(f, step_dir) for f in png_files]
            ccip_score_seq = ccip_metrics.score(png_files, mode='seq')
            ccip_score = ccip_score_seq.mean().item()
            aic_score_seq = aic_metrics.score(png_files, mode='seq')
            aic_score = aic_score_seq.mean().item()
            bp_score_seq = bp_metrics.score(png_files, mode='seq')
            bp_score = bp_score_seq.mean().item()
            logging.info(f'Step {step!r}, CCIP Score: {ccip_score:.4f}, '
                         f'AI-Corrupt Score: {aic_score:.4f}, Bikini Plus Score: {bp_score:.4f}')

            row = {'step': step, 'ccip': ccip_score, 'aic': aic_score, 'bp': bp_score}
            with open(step_metrics_file, 'w') as f:
                json.dump(row, f, indent=4, ensure_ascii=False)

            df = pd.DataFrame({
                'image': png_filenames,
                'ccip': ccip_score_seq,
                'aic': aic_score_seq,
                'bp': bp_score_seq,
            })
            df.to_csv(step_details_file, index=False)

        with open(step_metrics_file, 'r') as f:
            row = json.load(f)
        tb_data.append(row)

    df = pd.DataFrame(tb_data)
    metrics_plot_file = os.path.join(eval_dir, 'metrics_plot.png')
    logging.info(f'Plotting metrics to {metrics_plot_file!r} ...')
    df, df_selected = plt_metrics(
        df=df,
        model_name=meta_info['name'],
        dataset_size=meta_info['dataset']['size'],
        plot_file=metrics_plot_file,
        bs=meta_info['train']['dataset']['bs'],
        select=select,
        fidelity_alpha=fidelity_alpha,
    )

    metrics_csv_file = os.path.join(eval_dir, 'metrics.csv')
    logging.info(f'Save metrics table to {metrics_csv_file!r} ...')
    df.to_csv(metrics_csv_file, index=False)

    metrics_selected_csv_file = os.path.join(eval_dir, 'metrics_selected.csv')
    logging.info(f'Save selected metrics table to {metrics_selected_csv_file!r} ...')
    df_selected.to_csv(metrics_selected_csv_file, index=False)
