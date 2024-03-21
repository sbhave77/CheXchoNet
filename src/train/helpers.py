import pandas as pd
import numpy as np
import os
from visdom import Visdom
import skimage.exposure as hist
from os import makedirs
from os.path import exists, join
from datasets import zoom_2D
import cv2
import pydicom as dicom
import time
from pathlib import Path
from sklearn.metrics import confusion_matrix


def bootstrap_estimates(y_pred, y_true, metric, num_bootstraps=2000):
    boot_scores = []
    mean_score = metric(y_true, y_pred)

    for i in range(num_bootstraps):
        indices = np.random.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric(y_true[indices], y_pred[indices])
        boot_scores.append((score, (y_true[indices], y_pred[indices])))

    sorted_scores = sorted(boot_scores, key=lambda x: x[0])

    ci_lower, smallest_score_ys = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper, largest_score_ys = sorted_scores[int(0.975 * len(sorted_scores))]

    percentile_interval = (ci_lower, ci_upper)
    reverse_percentile_interval = (2 * mean_score - ci_upper, 2 * mean_score - ci_lower)

    return (
        percentile_interval,
        reverse_percentile_interval,
        smallest_score_ys,
        largest_score_ys,
    )


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name="main"):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel="Iterations",
                    ylabel=var_name,
                ),
            )
        else:
            self.viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                env=self.env,
                win=self.plots[var_name],
                name=split_name,
                update="append",
            )


def split_train_eval_test(metadata_df_path):
    df = pd.read_csv(metadata_df_path)

    pat_ids = list(set(df["patient_id"]))

    permuted_pat_ids = np.random.permutation(pat_ids)

    last_train_idx = int(len(permuted_pat_ids) * 0.90)

    last_eval_idx = int(
        last_train_idx + ((len(permuted_pat_ids) - last_train_idx) * 0.50)
    )

    train_pat_ids = permuted_pat_ids[:last_train_idx]
    eval_pat_ids = permuted_pat_ids[last_train_idx:last_eval_idx]
    test_pat_ids = permuted_pat_ids[last_eval_idx:]

    print("Num eval patids: {}".format(len(eval_pat_ids)))
    print("Num test patids: {}".format(len(test_pat_ids)))

    df_train = df[df["patient_id"].isin(train_pat_ids)]
    df_eval = df[df["patient_id"].isin(eval_pat_ids)]
    df_test = df[df["patient_id"].isin(test_pat_ids)]

    print("Eval SLVH PCT: {}".format(df_eval["slvh"].mean()))
    print("Eval DLV PCT: {}".format(df_eval["dlv"].mean()))
    print("Eval Composite PCT: {}".format(df_eval["composite_slvh_dlv"].mean()))

    print("Test SLVH PCT: {}".format(df_test["slvh"].mean()))
    print("Test DLV PCT: {}".format(df_test["dlv"].mean()))
    print("Test Composite PCT: {}".format(df_test["composite_slvh_dlv"].mean()))

    print(
        "Train pts/pct: {}, {} \n Eval pts/pct: {}, {} \n Test pts/pct: {}, {}".format(
            len(df_train),
            len(df_train) / float(len(df)),
            len(df_eval),
            len(df_eval) / float(len(df)),
            len(df_test),
            len(df_test) / float(len(df)),
        )
    )

    df_train.to_csv(
        metadata_df_path.replace(".csv", "_train_90_split.csv"), index=False
    )
    df_eval.to_csv(metadata_df_path.replace(".csv", "_eval_90_split.csv"), index=False)
    df_test.to_csv(metadata_df_path.replace(".csv", "_test_90_split.csv"), index=False)

    return df_train, df_eval, df_test


def update_train_stats(train_stats, new_mu, new_var, n_so_far, batch_size):

    mu, var = train_stats

    n = n_so_far
    m = float(batch_size)

    updated_mu = (n / (m + n)) * mu + (m / (m + n)) * new_mu

    updated_var = (
        (n / (m + n)) * var
        + (m / (m + n)) * new_var
        + ((m * n) / (m + n) ** 2) * (new_mu - mu) ** 2
    )

    train_stats[0] = updated_mu
    train_stats[1] = updated_var

    return train_stats
