import glob
import logging
import os.path
import shutil

import numpy as np
import pandas as pd
from hbutils.collection import unique
from hbutils.string import plural_word
from sdeval.controllability import BikiniPlusMetrics
from sdeval.corrupt import AICorruptMetrics
from sdeval.fidelity import CCIPMetrics
from tqdm import tqdm


def eval_for_infer_raw(workdir: str, pattern_top_n: int = 1, top_n: int = 30, fidelity_alpha: float = 2.0):
    infer_dir = os.path.join(workdir, 'infer')
    infer_raw_dir = os.path.join(infer_dir, 'raw')

    metrics_csv_file = os.path.join(infer_dir, 'metrics.csv')
    if not os.path.exists(metrics_csv_file):
        logging.info('Checking existing raw images ...')
        records = []
        for image_file in glob.glob(os.path.join(infer_raw_dir, '*', '*', '*.png')):
            rel_path = os.path.relpath(image_file, infer_raw_dir)
            step = int(os.path.basename(os.path.dirname(os.path.dirname(rel_path))))
            index = int(os.path.basename(os.path.dirname(rel_path)))
            pattern_name, ext = os.path.splitext(os.path.basename(rel_path))
            dst_filename = f'{step}_{index}_{pattern_name}{ext}'
            records.append({
                'file': image_file,
                'rel_file': rel_path,
                'dst_filename': dst_filename,
                'step': step,
                'index': index,
                'pattern': pattern_name,
            })

        df = pd.DataFrame(records)
        files = list(df['file'])
        features_path = os.path.join(workdir, 'features.npy')
        logging.info(f'Loading features from {features_path!r} ...')
        ccip_metrics = CCIPMetrics(None, feats=np.load(features_path))
        bp_metrics = BikiniPlusMetrics()
        aic_metrics = AICorruptMetrics()
        df['ccipd'] = ccip_metrics.score(files, mode='seq', algo='diff')
        df['aic'] = aic_metrics.score(files, mode='seq')
        df['bp'] = bp_metrics.score(files, mode='seq')
        df.to_csv(metrics_csv_file, index=False)
    else:
        df = pd.read_csv(metrics_csv_file)

    selected_csv_file = os.path.join(infer_dir, 'images_selected.csv')
    if not os.path.exists(selected_csv_file):
        logging.info(f'{plural_word(len(df), "image")} in total:\n{df}')
        df['ccipd_r'] = np.clip((df['ccipd'].mean() - df['ccipd']) / df['ccipd'].std() * 0.2 + 0.6, a_min=0.0,
                                a_max=1.0)
        df['bp_r'] = np.clip((df['bp'] - df['bp'].mean()) / df['bp'].std() * 0.2 + 0.6, a_min=0.0, a_max=1.0)
        df['integrate'] = (fidelity_alpha ** 2 + 1) * (df['ccipd_r'] * df['bp_r']) / \
                          (df['ccipd_r'] + fidelity_alpha ** 2 * df['bp_r'])

        aic_threshold = df['aic'].mean() - df['aic'].std() * 1.0
        ccipd_mean, ccipd_std = df['ccipd'].mean(), df['ccipd'].std()
        df = df[df['aic'] >= aic_threshold].sort_values(['integrate'], ascending=[False])
        selected_filenames = []
        for pattern in sorted(set(df['pattern'])):
            if 'pattern' in pattern:
                continue
            dfp = df[df['pattern'] == pattern]
            names = list(dfp[dfp['ccipd'] <= (ccipd_mean - ccipd_std * 1.35)]['dst_filename'])
            if len(names) < pattern_top_n:
                names.extend(
                    dfp[dfp['ccipd'] <= (ccipd_mean - ccipd_std * 0.5)][:pattern_top_n * 2]['dst_filename'])
                names = list(unique(names))
                if len(names) < pattern_top_n:
                    names.extend(
                        dfp[dfp['ccipd'] <= (ccipd_mean + ccipd_std * 0.5)][:pattern_top_n]['dst_filename'])
                    names = list(unique(names))
            selected_filenames.extend(names)
        selected_filenames.extend(df[:top_n]['dst_filename'])

        df_selected = df[df['dst_filename'].isin(selected_filenames)]
        logging.info(f'{plural_word(len(df_selected), "selected images")}:\n{df_selected}')

        selected_dir = os.path.join(infer_dir, 'selected')
        if os.path.exists(selected_dir):
            shutil.rmtree(selected_dir)
        os.makedirs(selected_dir, exist_ok=True)
        for item in tqdm(df_selected.to_dict('records'), desc='Copy Images'):
            src_file = os.path.join(infer_raw_dir, item['rel_file'])
            dst_file = os.path.join(selected_dir, item['dst_filename'])
            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
            shutil.copyfile(src_file, dst_file)
        df_selected.to_csv(selected_csv_file, index=False)
    else:
        logging.warning('Already selected, skipped.')
