import glob
import json
import logging
import os.path
from typing import Optional

import numpy as np
import pandas as pd
from imgutils.metrics import ccip_default_threshold
from matplotlib import pyplot as plt
from natsort import natsorted
from sdeval.controllability import BikiniPlusMetrics
from sdeval.corrupt import AICorruptMetrics
from sdeval.fidelity import CCIPMetrics
from tqdm import tqdm


def plt_metrics(df: pd.DataFrame, model_name: str, plot_file: str, select: int = 5, fidelity_alpha: float = 3.0,
                bp_std_min: Optional[float] = 0.015, min_aic: float = 0.8, aic_interval: float = 0.05,
                select_n: int = 3, ccip_distance_mode: bool = False,
                ccip_model: str = 'ccip-caformer-24-randaug-pruned'):
    # generate data calculation
    df = df.copy()

    ccip = np.array(df['ccip'])
    ccip_mean = ccip.mean()
    ccip_std = ccip.std()
    if ccip_distance_mode:
        ccip_norm = np.clip((-(ccip - ccip_mean) / ccip_std) * 0.2 + 0.6, a_min=1e-6, a_max=1.0)
    else:
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
    df = df[['step', 'epoch', 'filename', 'ccip', 'aic', 'bp', 'ccip_norm', 'bp_norm', 'integrate']]

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

    df_selected = df_aic.sort_values(
        by=['integrate', 'ccip', 'aic', 'bp'],
        ascending=[
            False,
            False if not ccip_distance_mode else True,
            False,
            False
        ]
    )[:select]
    selected_steps = df_selected['step'].tolist()
    aic_unselected_steps = sorted(set(aic_steps) - set(selected_steps))
    best_step = selected_steps[0]
    best_step_info = df_selected[df_selected['step'] == best_step].to_dict('records')[0]
    non_best_steps = sorted(set(selected_steps) - {best_step})

    # plot metrics
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(12, 9))

    axes[0, 0].plot(df['step'], df['ccip'])
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('CCIP Score' if not ccip_distance_mode else 'CCIP Distance')
    axes[0, 0].set_title('Fidelity' if not ccip_distance_mode else 'Difference')
    if ccip_distance_mode:
        ccip_threshold = ccip_default_threshold(model=ccip_model)
        axes[0, 0].axhspan(0.0, ccip_threshold * 0.3, facecolor='#dbf7c2')
        axes[0, 0].axhspan(ccip_threshold * 0.3, ccip_threshold * 0.6, facecolor='#f4f7c2')
        axes[0, 0].axhspan(ccip_threshold * 0.6, ccip_threshold, facecolor='#f7e4c2')
        axes[0, 0].axhspan(ccip_threshold, 1.0, facecolor='#f7c7c2')
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
                 f'Best Step: {best_step}, '
                 f'{"CCIP" if not ccip_distance_mode else "C-Dist"}: {best_step_info["ccip"]:.3f}, '
                 f'B-P: {best_step_info["bp"]:.3f}, '
                 f'AI-C: {best_step_info["aic"]:.3f}')
    plt.savefig(plot_file, dpi=150)
    plt.cla()

    return df, df_selected


def eval_for_workdir(workdir: str, select: Optional[int] = None, fidelity_alpha: float = 3.0,
                     ccip_distance_mode: bool = False, ccip_model: str = 'ccip-caformer-24-randaug-pruned',
                     **kwargs):
    from ..infer import find_steps_in_workdir, infer_with_workdir
    df_steps = find_steps_in_workdir(workdir)
    infer_with_workdir(workdir, **kwargs)

    logging.info(f'Evaluate for workdir {workdir!r} ...')
    with open(os.path.join(workdir, 'meta.json'), 'r') as f:
        meta_info = json.load(f)

    steps = list(df_steps['step'])
    logging.info(f'Steps {steps!r} found.')

    features_path = os.path.join(workdir, 'features.npy')
    logging.info(f'Loading features from {features_path!r} ...')
    ccip_metrics = CCIPMetrics(None, feats=np.load(features_path), model=ccip_model)
    bp_metrics = BikiniPlusMetrics()
    aic_metrics = AICorruptMetrics()

    current_eval_dir = os.path.join(workdir, 'eval')
    os.makedirs(current_eval_dir, exist_ok=True)
    tb_data = []
    for step_item in tqdm(df_steps.to_dict('records')):
        step, epoch, filename = step_item['step'], step_item['epoch'], step_item['filename']
        eval_dir = os.path.join(step_item['workdir'], 'eval')
        step_dir = os.path.join(eval_dir, str(step))
        os.makedirs(step_dir, exist_ok=True)
        step_metrics_file = os.path.join(step_dir, 'metrics.json')
        step_details_file = os.path.join(step_dir, 'details.csv')
        if not os.path.exists(step_details_file):
            png_files = natsorted(glob.glob(os.path.join(step_dir, '*.png')))
            png_filenames = [os.path.relpath(f, step_dir) for f in png_files]
            ccip_score_seq = ccip_metrics.score(
                png_files, algo='same' if not ccip_distance_mode else 'diff', mode='seq')
            ccip_score = ccip_score_seq.mean().item()
            aic_score_seq = aic_metrics.score(png_files, mode='seq')
            aic_score = aic_score_seq.mean().item()
            bp_score_seq = bp_metrics.score(png_files, mode='seq')
            bp_score = bp_score_seq.mean().item()
            logging.info(f'Step {step!r}, CCIP {"Score" if not ccip_distance_mode else "Distance"}: {ccip_score:.4f}, '
                         f'AI-Corrupt Score: {aic_score:.4f}, Bikini Plus Score: {bp_score:.4f}')

            row = {
                'step': step, 'epoch': epoch, 'filename': filename,
                'ccip': ccip_score, 'aic': aic_score, 'bp': bp_score,
            }
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
    metrics_plot_file = os.path.join(current_eval_dir, 'metrics_plot.png')
    logging.info(f'Plotting metrics to {metrics_plot_file!r} ...')
    select = select or max(min(len(steps) // 3, 5), 3)
    df, df_selected = plt_metrics(
        df=df,
        model_name=meta_info['name'],
        plot_file=metrics_plot_file,
        select=select,
        fidelity_alpha=fidelity_alpha,
        ccip_distance_mode=ccip_distance_mode,
    )

    metrics_csv_file = os.path.join(current_eval_dir, 'metrics.csv')
    logging.info(f'Save metrics table to {metrics_csv_file!r} ...')
    df.to_csv(metrics_csv_file, index=False)

    metrics_selected_csv_file = os.path.join(current_eval_dir, 'metrics_selected.csv')
    logging.info(f'Save selected metrics table to {metrics_selected_csv_file!r} ...')
    df_selected.to_csv(metrics_selected_csv_file, index=False)
    logging.info(f'Selected steps:\n{df_selected}\n')
