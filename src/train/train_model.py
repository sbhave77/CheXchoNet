import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score
from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import Timer, Checkpoint, DiskSaver, global_step_from_engine
from ignite.contrib.metrics import ROC_AUC, AveragePrecision
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Precision, Recall, RunningAverage, EpochWise, Average
from setup_metrics import (
    create_metrics,
    convert_continuous_dist,
    convert_continuous_to_binary,
)
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import sys
from helpers import VisdomLinePlotter, update_train_stats

sys.path.insert(0, "../model/")
from models import *
from libauc.losses import AUCMLoss, CrossEntropyLoss
from libauc.optimizers import PESG, Adam


def train_model(
    model,
    model_name,
    dataloader_train,
    dataloader_eval,
    dataloader_test,
    num_epochs,
    optimizer,
    optimizer_kwargs,
    class_to_name,
    class_to_name_continuous,
    input_size,
    num_labels,
    include_demo,
    continuous_labels,
    include_uncertainty,
    train_metrics_every_x_batches,
    eval_metrics_every_x_batches,
    label_to_weight,
    visdom_run_name,
    accumulate_grads_every_x_steps,
    use_auroc_loss,
    device,
    args,
):

    model = model.to(device)

    if use_auroc_loss:
        pos_weight = float(label_to_weight)
        imratio = 1.0 / (pos_weight + 1.0)
        print("IMRATIO: {}".format(imratio))
        lr = 0.05  # using smaller learning rate is better
        gamma = 500
        weight_decay = 1e-5
        margin = 1.0

        loss_fn = AUCMLoss(imratio=imratio)
        optimizer = PESG(
            model,
            a=loss_fn.a,
            b=loss_fn.b,
            alpha=loss_fn.alpha,
            imratio=imratio,
            lr=lr,
            gamma=gamma,
            margin=margin,
            weight_decay=weight_decay,
        )
    else:
        optimizer = optimizer(params=model.parameters(), **optimizer_kwargs)

    if continuous_labels:
        num_classes = len(class_to_name_continuous)
    else:
        num_classes = len(class_to_name)

    if include_demo:
        train_encoder = dataloader_train.dataset.enc
        train_age_stats = dataloader_train.dataset.age_stats

    if continuous_labels:
        if include_uncertainty:
            loss_fn = torch.distributions.Normal
        else:
            loss_fn = nn.MSELoss(reduction="none")
    else:
        label_to_weight = label_to_weight.to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=label_to_weight, reduction="none")

    args.loss_fn = loss_fn

    train_stats = [-1, -1]
    n_so_far = [0.0]

    def train_step(engine, batch):
        model.train()

        if include_demo:
            batch_images, batch_demo_info, batch_labels, batch_cont_labels = batch
        else:
            batch_images, batch_labels, batch_cont_labels = batch

        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        batch_cont_labels = batch_cont_labels.to(device)

        if include_demo:
            batch_demo_info = batch_demo_info.to(device)

        with torch.no_grad():
            mu, var = batch_images.mean(), batch_images.var()

        if engine.state.iteration == 1:
            train_stats[0] = mu
            train_stats[1] = var
            n_so_far[0] = n_so_far[0] + batch_images.shape[0]
        elif engine.state.epoch == 1:
            update_train_stats(train_stats, mu, var, n_so_far[0], batch_images.shape[0])
            n_so_far[0] = n_so_far[0] + batch_images.shape[0]

        batch_images = (batch_images - train_stats[0]) / torch.sqrt(
            train_stats[1] + 1e-5
        )

        if include_demo:
            outputs = model.forward(batch_images, batch_demo_info)
        else:
            outputs = model.forward(batch_images)

        if continuous_labels and include_uncertainty:
            mean_indexes = outputs.shape[1] // 2
            outputs = torch.nn.functional.softplus(outputs)
            normal_dist = loss_fn(outputs[:, :mean_indexes], outputs[:, mean_indexes:])
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

        loss.backward()

        if (i + 1) % accumulate_grads_every_x_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        ret_items = {
            "loss": loss.item(),
            "loss_per_class": loss_per_class,
            "y_true": batch_labels.float(),
        }

        if not continuous_labels:
            probs = torch.sigmoid(outputs)
            ret_items["y_prob"] = probs
        else:
            ret_items["cont_preds"] = outputs[:, :mean_indexes]
            ret_items["cont_true"] = batch_cont_labels.float()

        return ret_items

    def eval_step(engine, batch):
        model.eval()

        with torch.no_grad():

            if include_demo:
                batch_images, batch_demo_info, batch_labels, batch_cont_labels = batch
            else:
                batch_images, batch_labels, batch_cont_labels = batch

            for p in model.parameters():
                p.grad = None

            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            batch_cont_labels = batch_cont_labels.to(device)

            if include_demo:
                batch_demo_info = batch_demo_info.to(device)

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

            if not continuous_labels:
                probs = torch.sigmoid(outputs)
                ret_items["y_prob"] = probs
            else:
                ret_items["cont_preds"] = outputs[:, :mean_indexes]
                ret_items["cont_true"] = batch_cont_labels.float()

            return ret_items

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)

    train_metrics = create_metrics(num_classes, continuous_labels)
    eval_metrics = create_metrics(num_classes, continuous_labels, is_eval=True)

    for metric_name, mt in train_metrics.items():
        if metric_name in ["avg_loss", "running_avg_loss"]:
            mt.attach(engine=trainer, name=metric_name)
        else:
            for i in range(num_classes):
                mt[i].attach(engine=trainer, name=metric_name + "_" + str(i))

    for metric_name, mt in eval_metrics.items():
        if metric_name in ["avg_loss", "running_avg_loss"]:
            mt.attach(
                engine=evaluator,
                name=metric_name,
            )
        else:
            for i in range(num_classes):
                mt[i].attach(engine=evaluator, name=metric_name + "_" + str(i))

    vis = VisdomLinePlotter(env_name=visdom_run_name)

    @trainer.on(
        Events.ITERATION_COMPLETED(every=train_metrics_every_x_batches)
        | Events.EPOCH_COMPLETED
    )
    def log_training_statistics(engine):
        metrics = trainer.state.metrics
        batch_loss = trainer.state.output["loss"]
        running_avg_loss_per_class = []

        metric_numbers = defaultdict(list)

        for metric_name, mt in train_metrics.items():
            if type(mt) == list:
                for m in mt:
                    try:
                        metric_numbers[metric_name].append(float(m.compute()))
                    except:
                        metric_numbers[metric_name].append(0)
            else:
                metric_numbers[metric_name].append(float(mt.compute()))

        if not continuous_labels:
            met_list = [
                (
                    "AUROC",
                    "Training AUROC (reset every epoch)",
                    metric_numbers["roc_auc_metric"],
                ),
                (
                    "AURPC",
                    "Training AUPRC (reset every epoch)",
                    metric_numbers["precision_recall_metric"],
                ),
                (
                    "Precision",
                    "Training Precision (reset every epoch)",
                    metric_numbers["precision_metric"],
                ),
                (
                    "Recall",
                    "Training Recall (reset every epoch)",
                    metric_numbers["recall_metric"],
                ),
                (
                    "Running Avg Loss Per Class",
                    "Running Avg Loss per class",
                    metric_numbers["running_avg_loss_per_class"],
                ),
            ]
        else:
            met_list = [
                (
                    "Running Avg Loss Per Class",
                    "Training Running Avg Loss per class",
                    metric_numbers["running_avg_loss_per_class"],
                ),
                ("R2 Score", "Training R2 Score per class", metric_numbers["r2_score"]),
                (
                    "RMSE Score",
                    "Training RMSE Score per class",
                    metric_numbers["rmse_score"],
                ),
            ]

        for plot_id, plot_title, value_list in met_list:
            for c in range(num_classes):
                if continuous_labels:
                    vis.plot(
                        plot_id,
                        class_to_name_continuous[c],
                        plot_title,
                        engine.state.iteration,
                        value_list[c],
                    )
                else:
                    vis.plot(
                        plot_id,
                        class_to_name[c],
                        plot_title,
                        engine.state.iteration,
                        value_list[c],
                    )

        vis.plot(
            "Batch Loss Running Avg",
            "loss",
            "Training Batch Loss Running Average",
            engine.state.iteration,
            metric_numbers["running_avg_loss"][0],
        )

        vis.viz.save([visdom_run_name])

    @trainer.on(
        Events.ITERATION_COMPLETED(every=eval_metrics_every_x_batches)
        | Events.EPOCH_COMPLETED
    )
    def log_evaluation_statistics(engine):
        evaluator.run(dataloader_eval)
        metrics = evaluator.state.metrics
        batch_loss = evaluator.state.output["loss"]

        if not continuous_labels:
            auroc = [
                metrics["roc_auc_metric" + "_" + str(i)] for i in range(num_classes)
            ]
            auprc = [
                metrics["precision_recall_metric" + "_" + str(i)]
                for i in range(num_classes)
            ]
            prec = [
                metrics["precision_metric" + "_" + str(i)] for i in range(num_classes)
            ]
            rec = [metrics["recall_metric" + "_" + str(i)] for i in range(num_classes)]
        else:
            r2_score = [metrics["r2_score" + "_" + str(i)] for i in range(num_classes)]
            rmse_score = [
                metrics["rmse_score" + "_" + str(i)] for i in range(num_classes)
            ]

        avg_loss_per_class = [
            metrics["avg_loss_per_class" + "_" + str(i)] for i in range(num_classes)
        ]
        avg_loss = metrics["avg_loss"]

        if not continuous_labels:
            met_list = [
                ("Eval AUROC", "Eval AUROC (reset every epoch)", auroc),
                ("Eval AUPRC", "Eval AUPRC (reset every epoch)", auprc),
                ("Eval Precision", "Eval Precision (reset every epoch)", prec),
                ("Eval Recall", "Eval Recall (reset every epoch)", rec),
                (
                    "Eval Avg Loss Per Class",
                    "Eval Avg Loss per class",
                    avg_loss_per_class,
                ),
            ]
        else:
            met_list = [
                (
                    "Eval Avg Loss Per Class",
                    "Eval Avg Loss per class",
                    avg_loss_per_class,
                ),
                ("Eval R2 Score", "Eval R2 Score per class", r2_score),
                ("Eval RMSE Score", "Eval RMSE Score per class", rmse_score),
            ]

        for plot_id, plot_title, value_list in met_list:
            for c in range(num_classes):
                if continuous_labels:
                    vis.plot(
                        plot_id,
                        class_to_name_continuous[c],
                        plot_title,
                        engine.state.iteration,
                        value_list[c],
                    )
                else:
                    vis.plot(
                        plot_id,
                        class_to_name[c],
                        plot_title,
                        engine.state.iteration,
                        value_list[c],
                    )

        vis.plot(
            "Average Loss",
            "loss",
            "Eval Average Loss",
            engine.state.iteration,
            avg_loss,
        )
        vis.viz.save([visdom_run_name])

    def loss_score_function(engine):
        return -1 * engine.state.metrics["avg_loss"]

    def auroc_score_function(engine):
        return (
            torch.tensor(
                [
                    engine.state.metrics["roc_auc_metric" + "_" + str(i)]
                    for i in range(num_classes)
                ]
            )
            .mean()
            .item()
        )

    best_models_path = Path("./best_models", visdom_run_name)
    best_models_path.mkdir(parents=True, exist_ok=True)

    loss_to_save = {"model": model, "opt": optimizer, "trainer": trainer}

    @trainer.on(Events.EPOCH_COMPLETED(once=1))
    def log_dataloader_stats(engine):
        if include_demo:
            torch.save(
                {
                    "dataloader_test": dataloader_test,
                    "train_stats": train_stats,
                    "train_encoder": train_encoder,
                    "train_age_stats": train_age_stats,
                    "args": args,
                },
                os.path.join(best_models_path, "test_set_settings.pth"),
            )
        else:
            torch.save(
                {
                    "dataloader_test": dataloader_test,
                    "train_stats": train_stats,
                    "args": args,
                },
                os.path.join(best_models_path, "test_set_settings.pth"),
            )

    checkpoint_loss_handler = Checkpoint(
        loss_to_save,
        DiskSaver(best_models_path, create_dir=True),
        n_saved=4,
        filename_prefix="best",
        score_function=loss_score_function,
        score_name="val_loss",
        global_step_transform=global_step_from_engine(trainer),
    )

    evaluator.add_event_handler(Events.COMPLETED, checkpoint_loss_handler)

    auroc_to_save = {"model": model, "opt": optimizer, "trainer": trainer}

    if not continuous_labels:

        checkpoint_auroc_handler = Checkpoint(
            auroc_to_save,
            DiskSaver(best_models_path, create_dir=True),
            n_saved=4,
            filename_prefix="best",
            score_function=auroc_score_function,
            score_name="auroc",
            global_step_transform=global_step_from_engine(trainer),
        )

        evaluator.add_event_handler(Events.COMPLETED, checkpoint_auroc_handler)

    pbar = ProgressBar()
    pbar.attach(trainer)

    trainer.run(dataloader_train, max_epochs=num_epochs)
