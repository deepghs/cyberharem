import glob
import logging
import os.path

import numpy as np
import pandas as pd
from hbutils.string import plural_word
from sdeval.controllability import BikiniPlusMetrics
from sdeval.corrupt import AICorruptMetrics
from sdeval.fidelity import CCIPMetrics


def eval_for_infer_raw(workdir: str):
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

    logging.info(f'{plural_word(len(df), "image")} in total:\n{df}')
    return df
