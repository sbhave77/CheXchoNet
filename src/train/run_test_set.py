import argparse
import torch
import torch.nn as nn
import pickle
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import Timer, Checkpoint, DiskSaver, global_step_from_engine
from ignite.contrib.metrics import ROC_AUC, AveragePrecision
from ignite.metrics import Precision, Recall, RunningAverage, EpochWise, Average
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from setup_metrics import (
    convert_continuous_dist,
    convert_continuous_to_binary,
    create_metrics,
)
from pathlib import Path
import os
import sys
from helpers import VisdomLinePlotter, bootstrap_estimates

sys.path.insert(0, "../model/")
from models import (
    DenseNet121Base,
    DenseNet121MultiView,
    EfficientNetDynamic,
    DenseNet121BaseWithDemo,
)
import numpy as np
from datasets import CXRDataset, collate_cxr, StanfordCXRDataset
from torch.utils.data import DataLoader


def run_test_set(checkpoint_fp, test_set_settings_path, file_name=""):

    checkpoint = torch.load(checkpoint_fp, map_location=torch.device("cuda:3"))
    tst = torch.load(test_set_settings_path, map_location=torch.device("cuda:3"))

    include_demo = tst["args"].include_demo

    if include_demo:
        dataloader_test, train_stats, train_encoder, train_age_stats, args = (
            tst["dataloader_test"],
            tst["train_stats"],
            tst["train_encoder"],
            tst["train_age_stats"],
            tst["args"],
        )
    else:
        dataloader_test, train_stats, args = (
            tst["dataloader_test"],
            tst["train_stats"],
            tst["args"],
        )

    args.device = torch.device("cuda:3")
    train_stats = [t.to(args.device) for t in train_stats]

    load = {"model": None}
    if args.model_name == "dense_base":
        if args.continuous_labels and args.include_uncertainty:
            model = DenseNet121Base(args.num_labels * 2, model_kwargs=args.model_kwargs)
        else:
            model = DenseNet121Base(args.num_labels, model_kwargs=args.model_kwargs)
        load["model"] = model
    elif args.model_name == "pa_ll":
        model = DenseNet121MultiView(args.num_labels)
        load["model"] = model
    elif args.model_name == "efficient_net":
        model = EfficientNetDynamic(args.num_labels, args.input_size, pretrained=True)
        load["model"] = model
    elif args.model_name == "dense_base_with_demo":
        if args.continuous_labels and args.include_uncertainty:
            model = DenseNet121BaseWithDemo(
                args.num_labels * 2,
                demo_size=args.demo_size,
                model_kwargs=args.model_kwargs,
            )
        else:
            model = DenseNet121BaseWithDemo(
                args.num_labels,
                demo_size=args.demo_size,
                model_kwargs=args.model_kwargs,
            )
        load["model"] = model

    Checkpoint.load_objects(load, checkpoint)

    model = model.to(args.device)

    dataset_test = dataloader_test.dataset

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_cxr,
        pin_memory=True,
        num_workers=16,
    )

    return calculate_test_statistics(
        model,
        dataloader_test,
        args.num_labels,
        args.class_to_name,
        args.class_to_name_continuous,
        args.device,
        train_stats,
        args.include_demo,
        args.continuous_labels,
        args.include_uncertainty,
        args.loss_fn,
        file_name,
    )


def calculate_test_statistics(
    model,
    test_dloader,
    num_classes,
    class_to_name,
    class_to_name_continuous,
    device,
    train_stats,
    include_demo,
    continuous_labels,
    include_uncertainty,
    loss_fn,
    file_name,
):

    model.to(device)

    # evaluate binary labels
    num_classes = len(class_to_name)

    train_encoder = test_dloader.dataset.enc

    def test_step(engine, batch):
        model.eval()

        with torch.no_grad():

            if include_demo:
                batch_images, batch_demo_info, batch_labels, batch_cont_labels = batch
                batch_demo_info = batch_demo_info.to(device)
            else:
                batch_images, batch_labels, batch_cont_labels = batch

            for p in model.parameters():
                p.grad = None

            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            batch_cont_labels = batch_cont_labels.to(device)

            batch_images = (batch_images - train_stats[0]) / (
                torch.sqrt(train_stats[1]) + 1e-5
            )

            if include_demo:
                outputs = model.forward(batch_images, batch_demo_info)
            else:
                outputs = model.forward(batch_images)

            if continuous_labels and include_uncertainty:
                mean_indexes = outputs.shape[1] // 2
                outputs = torch.nn.functional.softplus(outputs)
                normal_dist = loss_fn(
                    outputs[:, :mean_indexes], outputs[:, mean_indexes:]
                )
                nll = -1 * normal_dist.log_prob(batch_cont_labels.float())
                loss_per_class = nll.mean(axis=0)
                loss = loss_per_class.mean()
            elif continuous_labels and not include_uncertainty:
                loss_unreduced = loss_fn(outputs, batch_cont_labels.float())
                loss_per_class = loss_unreduced.mean(axis=0)
                loss = loss_unreduced.mean()
            else:
                loss_unreduced = loss_fn(outputs, batch_labels.float())
                loss_per_class = loss_unreduced.mean(axis=0)
                loss = loss_unreduced.mean()

            ret_items = {
                "loss": loss.item(),
                "loss_per_class": loss_per_class,
                "y_true": batch_labels.float(),
            }

            if include_demo:
                ret_items["demo_info"] = batch_demo_info

            if continuous_labels:
                if include_uncertainty:
                    ret_items["normal_dist"] = normal_dist
                    ret_probs, ret_preds = convert_continuous_dist(
                        ret_items,
                        class_to_name_continuous,
                        train_encoder,
                        class_to_name,
                    )
                    ret_items["y_prob"] = ret_probs
                    ret_items["y_pred"] = ret_preds
                    ret_items["cont_preds"] = outputs[:, :mean_indexes]
                    ret_items["cont_true"] = batch_cont_labels.float()
                else:
                    out_probs = convert_continuous_to_binary(
                        ret_items, class_to_name_continuous, train_encoder
                    )
                    ret_items["y_prob"] = out_probs

            if not continuous_labels:
                probs = torch.sigmoid(outputs)
                ret_items["y_prob"] = probs

            return ret_items

    tester = Engine(test_step)

    test_metrics = create_metrics(
        len(class_to_name), continuous_labels=False, is_eval=True, include_loss=False
    )

    for metric_name, mt in test_metrics.items():
        if metric_name in ["avg_loss", "running_avg_loss"]:
            mt.attach(engine=tester, name=metric_name, usage=EpochWise.usage_name)
        else:
            for i in range(len(class_to_name)):
                mt[i].attach(
                    engine=tester,
                    name=metric_name + "_" + str(i),
                    usage=EpochWise.usage_name,
                )

    all_preds = {}
    all_targets = {}

    @tester.on(Events.EPOCH_COMPLETED)
    def log_test_statistics(engine):
        metrics = tester.state.metrics

        auroc = [metrics["roc_auc_metric" + "_" + str(i)] for i in range(num_classes)]
        auprc = [
            metrics["precision_recall_metric" + "_" + str(i)]
            for i in range(num_classes)
        ]
        prec = [metrics["precision_metric" + "_" + str(i)] for i in range(num_classes)]
        rec = [metrics["recall_metric" + "_" + str(i)] for i in range(num_classes)]

        roc_auc_metric_test = test_metrics["roc_auc_metric"]

        roc_ci = []
        pr_ci = []
        conf_matrix = []
        save_preds = []
        save_targets = []
        for i in range(num_classes):
            preds, targets = torch.cat(roc_auc_metric_test[i]._predictions), torch.cat(
                roc_auc_metric_test[i]._targets
            )

            save_preds.append(preds)
            save_targets.append(targets)

            precision, recall, thresholds = precision_recall_curve(targets, preds)
            f1_scores = 2 * recall * precision / (recall + precision)
            best_thresh = thresholds[np.argmax(f1_scores)]
            print(best_thresh)

            c_mat = confusion_matrix(targets, preds > 0.5)
            conf_matrix.append((c_mat, 0.5))

            _, roc_ci_int, _, _ = bootstrap_estimates(preds, targets, roc_auc_score)
            _, pr_ci_int, _, _ = bootstrap_estimates(
                preds, targets, average_precision_score
            )

            all_preds[i] = preds
            all_targets[i] = targets

            roc_ci.append(roc_ci_int)
            pr_ci.append(pr_ci_int)

        ret_metrics = {}
        for i in range(num_classes):

            ret_metrics[class_to_name[i]] = {
                "auroc": (auroc[i], (roc_ci[i][0], roc_ci[i][1])),
                "auprc": (auprc[i], (pr_ci[i][0], pr_ci[i][1])),
            }

        df = test_dloader.dataset.metadata
        df = df.sort_index()

        try:
            for i in range(num_classes):
                df[class_to_name[i] + "_preds"] = save_preds[i]
                df[class_to_name[i] + "_labs"] = save_targets[i]
        except:
            import pdb

            pdb.set_trace()

        df.to_csv(f"metadata_with_predictions_{file_name}_updated.csv", index=False)

        label_true = [c + "_labs" for c in class_to_name]
        label_preds = [c + "_preds" for c in class_to_name]

        labels_zipped = list(zip(class_to_name, list(zip(label_true, label_preds))))

        with open("ret_mets_{}_updated.pkl".format(file_name), "wb") as wb:
            pickle.dump({"ret_met": ret_metrics}, wb)

        for i in range(len(class_to_name)):
            print(
                "\n--------------------------------------------------------------------------------------------------"
                "***TEST SET*** CLASS_NAME: {} ## AUROC: {} ({},{}) ## AUPRC: {} ({},{}) ## PREC: {} ## RECALL: {}"  ## FPR/TPR: ({}, {})"
                "--------------------------------------------------------------------------------------------------\n".format(
                    class_to_name[i],
                    auroc[i],
                    roc_ci[i][0],
                    roc_ci[i][1],
                    auprc[i],
                    pr_ci[i][0],
                    pr_ci[i][1],
                    prec[i],
                    rec[i],
                )
            )

    pbar = ProgressBar()
    pbar.attach(tester)

    tester.run(test_dloader, max_epochs=1)

    class_to_name_dict = {}
    for i, v in enumerate(class_to_name):
        class_to_name_dict[i] = v

    return all_preds, all_targets


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-ckpt", "--ckpt_path", type=str, required=True, dest="checkpoint_fp"
    )

    parser.add_argument(
        "-ts",
        "--test_set_settings_path",
        type=str,
        required=True,
        dest="test_set_settings_path",
    )

    parser.add_argument(
        "-fname" "--file_name", type=str, required=False, dest="file_name"
    )

    args_dict = vars(parser.parse_args())

    run_test_set(**args_dict)
