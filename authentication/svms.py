'''
CDP authentication with SVM using previously calculated metrics by metrics.py script
Author: Roman CHABAN, University of Geneva, 2021
'''

import argparse
from pathlib import Path
from itertools import product

import p_tqdm
import numpy as np
import pandas as pd

from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser()
parser.add_argument("input_csv", default='metrics.csv', type=Path, help="The path to the csv file which is the output of metrics.py")
parser.add_argument("--cpus", default=6, type=int, help="Number of CPUs used for parallelization")
args = parser.parse_args()


unfold = lambda x: [item for sublist in x for item in sublist]
calc = lambda x: unfold(p_tqdm.p_map(calc_single, x, num_cpus=args.cpus))
toperc = lambda x: round(x * 100, 2)

blobs = {
    'HPI76_des3_812.8dpi_2400dpi': r'$x^{76}$',
    'HPI55_des3_812.8dpi_2400dpi': r'$x^{55}$',
    'HPI76_EHPI55_des3_812.8dpi_2400dpi': r'$f^{76/55}$',
    'HPI76_EHPI76_des3_812.8dpi_2400dpi': r'$f^{76/76}$',
    'HPI55_EHPI55_des3_812.8dpi_2400dpi': r'$f^{55/55}$',
    'HPI55_EHPI76_des3_812.8dpi_2400dpi': r'$f^{55/76}$',
}

# Here you can specify the list of subsets used for one-class SVM training.
# At the moment there are 2 items with 1 subset each. 
# In this scenario there will be trained 2 models. Each model is trained on 1 subset.
train_subsets_1 = [
    [
        'HPI55_des3_812.8dpi_2400dpi',
    ],
    [
        'HPI76_des3_812.8dpi_2400dpi',
    ]
]


# The same logic as above applise here as well. But it is for two-class SVM.
train_subsets_2 = [
    [
        'HPI76_des3_812.8dpi_2400dpi',
        'HPI76_EHPI55_des3_812.8dpi_2400dpi',
    ],
    [
        'HPI76_des3_812.8dpi_2400dpi',
        'HPI76_EHPI76_des3_812.8dpi_2400dpi',
    ],
    [
        'HPI55_des3_812.8dpi_2400dpi',
        'HPI55_EHPI55_des3_812.8dpi_2400dpi',
    ],
    [
        'HPI55_des3_812.8dpi_2400dpi',
        'HPI55_EHPI76_des3_812.8dpi_2400dpi',
    ]
]

# The subsets used for the model testing.
test_subsets = {
    'HPI55': [
        'HPI55_des3_812.8dpi_2400dpi',
        'HPI55_EHPI55_des3_812.8dpi_2400dpi',
        'HPI55_EHPI76_des3_812.8dpi_2400dpi'
    ],
    'HPI76': [
        'HPI76_des3_812.8dpi_2400dpi',
        'HPI76_EHPI55_des3_812.8dpi_2400dpi',
        'HPI76_EHPI76_des3_812.8dpi_2400dpi'
    ]
}


def calc_single(unpack):
    # Passing parameters like this because of Python's limitation in terms of parallelization.
    mode, train_subset, ratio, seed, set_metrics = unpack

    printer_defender = train_subset[0][:5]

    # Preparing train data
    df_train = df_in[df_in.subset.isin(train_subset)]
    X = df_train[list(set_metrics)].to_numpy()
    Y = df_train.Y.to_numpy()
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=1 - ratio, random_state=seed)

    # Creating and fitting model. If the variable 'mode' contains 1 then one-class SVM is created.
    # If 'mode' contains 2 - then two-class
    if mode.find('2') != -1:
        model = make_pipeline(StandardScaler(), SVC(gamma=0.3, kernel='rbf'))
        model.fit(X_train, Y_train)
    elif mode.find('1') != -1:
        model = make_pipeline(StandardScaler(), OneClassSVM(nu=0.01, gamma=0.3, kernel='rbf'))
        model.fit(X_train)

    results_local = []

    # Testing model on the testing subsets
    for printer_attacker, printer_test_subset in test_subsets.items():

        for test_subset in printer_test_subset:
            df_test = df_in[(df_in.subset == test_subset)]
            X_test = df_test[list(set_metrics)].to_numpy()
            Y_test = df_test.Y.to_numpy()
            Y_pred = model.predict(X_test)
            Y_pred[Y_pred == -1] = 0
            acc_val = 1 - accuracy_score(Y_test, Y_pred)

            # If subset is original then we accumulate probability of false acceptance
            # Otherwise - probability of false miss
            if test_subset.find('EHPI') == -1:
                pmiss = acc_val
                pfa = 0
            else:
                pfa = acc_val
                pmiss = 0

            results_local.append([
                mode, ', '.join([blobs[t] for t in train_subset]), blobs[test_subset], printer_defender, printer_attacker, 
                ratio, seed, ', '.join(list(set_metrics)), pmiss, pfa
            ])
    return results_local


if __name__ == '__main__':
    df_in = pd.read_csv(args.input_csv, index_col=0)
    df_in['Y'] = (df_in.subset.str.find('EHPI') == -1).astype(int)

    # Here can be specified various sets of ratios, seeds and set of all metrics
    ratios = [0.2]
    seeds = list(range(20))
    metrics = ['ssim', 'corr', 'jaccard', 'hamming']

    svm1_results = calc(list(product(['svm1a'], train_subsets_1, ratios, seeds, [metrics])))
    svm1p_results = calc(list(product(['svm1p'], train_subsets_1, ratios, seeds, [['corr', 'jaccard'], ['hamming', 'ssim']])))

    svm2_results = calc(list(product(['svm2a'], train_subsets_2, ratios, seeds, [metrics])))
    svm2p_results = calc(list(product(['smv2p'], train_subsets_2, ratios, seeds, [['corr', 'jaccard'], ['hamming', 'ssim']])))

    df_out = pd.DataFrame(svm1_results + svm1p_results + svm2_results + svm2p_results)
    df_out = df_out.set_axis(
        [
            'Mode', 'Trained', 'Tested', r'$P_D$', r'$P_A$', 'Ratio', 'Seed', 'Metrics', r'$P_{amiss}$', r'$P_{fa}$'
        ],
        axis='columns', inplace=False)

    df_out.to_csv('svms.csv')
