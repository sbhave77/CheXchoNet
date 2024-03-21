import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
)
from collections import defaultdict
from tqdm import tqdm
import sys
from helpers import bootstrap_estimates

def sample_single_cxr_per_person(metadata_with_preds, labels_zipped, num_rand_samples=20, direct_df=False, pat_id="PatientID"):

    if direct_df:
        df = metadata_with_preds
    else:
        df = pd.read_csv(metadata_with_preds)

    boot_auroc = defaultdict(list)
    boot_auprc = defaultdict(list)

    for i in tqdm(range(num_rand_samples)):
        
        rand_sample_by_pat_id = df.groupby(pat_id).apply(lambda x: x.sample(1)).reset_index(drop=True)

        for lab, (label_col, pred_col) in labels_zipped:

            auroc_score = roc_auc_score(rand_sample_by_pat_id[label_col], rand_sample_by_pat_id[pred_col])
            auprc_score = average_precision_score(rand_sample_by_pat_id[label_col], rand_sample_by_pat_id[pred_col])

            boot_auroc[lab].append(auroc_score)
            boot_auprc[lab].append(auprc_score)

    ret_metrics = {}

    for lab in boot_auroc.keys():

        auroc_mean, auroc_2sd = np.array(boot_auroc[lab]).mean(), 1.96 * np.array(boot_auroc[lab]).std()
        auprc_mean, auprc_2sd = np.array(boot_auprc[lab]).mean(), 1.96 * np.array(boot_auprc[lab]).std()        

        print(
            "\n--------------------------------------------------------------------------------------------------"
            "***TEST SET*** CLASS_NAME: {} ## AUROC: {} ({},{}) ## AUPRC: {} ({},{})"
            "--------------------------------------------------------------------------------------------------\n"
            .format(lab, auroc_mean, auroc_mean - auroc_2sd, auroc_mean + auroc_2sd,
                    auprc_mean, auprc_mean - auprc_2sd, auprc_mean + auprc_2sd))

        ret_metrics[lab] = {
            "auroc": (auroc_mean, (auroc_mean - auroc_2sd, auroc_mean + auroc_2sd)),
            "auprc": (auprc_mean, (auprc_mean - auprc_2sd, auprc_mean + auprc_2sd))
        }

    return ret_metrics

def get_metrics_from_metadata(metadata_with_preds, labels_zipped, direct_df=False):

    if direct_df:
        df = metadata_with_preds
    else:
        df = pd.read_csv(metadata_with_preds)

    ret_metrics = {}

    for lab, (label_col, pred_col) in labels_zipped:
        
        auroc_score = roc_auc_score(df[label_col], df[pred_col])
        auprc_score = average_precision_score(df[label_col], df[pred_col])

        _, roc_ci, _, _ = bootstrap_estimates(df[pred_col].to_numpy(), df[label_col].to_numpy(), roc_auc_score)
        _, pr_ci, _, _ = bootstrap_estimates(df[pred_col].to_numpy(), df[label_col].to_numpy(), average_precision_score)

        ret_metrics[lab] = {
            "auroc": (auroc_score, roc_ci),
            "auprc": (auprc_score, pr_ci)
        }

    return ret_metrics
