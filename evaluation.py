import json
import argparse
import os
from importlib import import_module

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


def evaluation(gt_dir, pred_dir):
    """
    Args:
        gt_dir (string) : root directory of ground truth file
        pred_dir (string) : root directory of prediction file (output of inference.py)
    """
    num_classes = 18
    results = {}

    gt = pd.read_csv(os.path.join(gt_dir, 'gt_answer.csv'))
    pred = pd.read_csv(os.path.join(pred_dir, 'output_efficientnet_b4_single_lr1e-5_mask.csv'))
    cls_report = classification_report(gt.ans.values, pred.ans.values, labels=np.arange(num_classes), output_dict=True, zero_division=0)
    acc = cls_report['accuracy'] * 100
    f1 = np.mean([cls_report[str(i)]['f1-score'] for i in range(num_classes)])

    results['accuracy'] = {
        'value': f'{acc:.2f}%',
        'rank': True,
        'decs': True,
    }
    results['f1'] = {
        'value': f'{f1:.2f}%',
        'rank': False,
        'decs': True,
    }

    return json.dumps(results)

if __name__ == '__main__':
   gt_dir = '/opt/ml/code'
   pred_dir = '/opt/ml/code/level1-image-classification-level1-cv-01/output'

   from pprint import pprint
   results = evaluation(gt_dir, pred_dir)
   pprint(results)
