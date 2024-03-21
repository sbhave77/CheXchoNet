from thresholds import THRESHOLDS
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from ignite.metrics import (
    Precision,
    Recall,
    RunningAverage,
    BatchWise,
    EpochWise,
    Average,
    RootMeanSquaredError,
)
from ignite.contrib.metrics import ROC_AUC, AveragePrecision
from ignite.contrib.metrics.regression import R2Score
from custom_metrics import SpecAndSens
import torch


def create_metrics(
    num_classes, continuous_labels=False, is_eval=False, include_loss=True
):

    def create_output_func(i, rd=False, preds=False):

        def transform_func(x):
            return x["y_prob"][:, i], x["y_true"][:, i]

        def transform_func_rd(x):
            return x["y_prob"][:, i].round(), x["y_true"][:, i]

        def transform_func_preds(x):
            return x["y_pred"][:, i], x["y_true"][:, i]

        def transform_func_cont(x):
            return x["cont_preds"][:, i], x["cont_true"][:, i]

        if continuous_labels:
            return transform_func_cont
        else:
            if rd:
                return transform_func_rd
            elif preds:
                return transform_func_preds
            else:
                return transform_func

    def create_loss_per_class(j):

        def loss_per_class(x):
            return x["loss_per_class"][j]

        return loss_per_class

    all_metrics = {}

    if not continuous_labels:

        all_metrics["roc_auc_metric"] = [
            ROC_AUC(output_transform=create_output_func(j)) for j in range(num_classes)
        ]
        all_metrics["precision_recall_metric"] = [
            AveragePrecision(output_transform=create_output_func(j))
            for j in range(num_classes)
        ]

        all_metrics["precision_metric"] = [
            Precision(output_transform=create_output_func(j, True))
            for j in range(num_classes)
        ]
        all_metrics["recall_metric"] = [
            Recall(output_transform=create_output_func(j, True))
            for j in range(num_classes)
        ]
    else:
        all_metrics["r2_score"] = [
            R2Score(output_transform=create_output_func(j, False))
            for j in range(num_classes)
        ]
        all_metrics["rmse_score"] = [
            RootMeanSquaredError(output_transform=create_output_func(j, False))
            for j in range(num_classes)
        ]

    if include_loss:
        if is_eval:
            all_metrics["avg_loss"] = Average(output_transform=lambda x: x["loss"])
            all_metrics["avg_loss_per_class"] = [
                Average(output_transform=create_loss_per_class(i))
                for i in range(num_classes)
            ]
        else:
            all_metrics["running_avg_loss"] = RunningAverage(
                output_transform=lambda x: x["loss"]
            )
            all_metrics["running_avg_loss_per_class"] = [
                RunningAverage(output_transform=create_loss_per_class(i))
                for i in range(num_classes)
            ]

    return all_metrics


def convert_continuous_dist(x, continuous_label_names, encoder, binary_labels):
    demo_size = 2
    sex_info = x["demo_info"][:, :demo_size]
    sex = encoder.inverse_transform(sex_info.cpu())
    categories_in_order = encoder.categories_[0].tolist()
    m_index, f_index = categories_in_order.index("M"), categories_in_order.index("F")

    if "slvh" in binary_labels:
        assert "ivsd" in continuous_label_names
        assert "lvpwd" in continuous_label_names

    if "dlv" in binary_labels:
        assert "lvidd" in continuous_label_names

    batch_size, num_final_labels, num_cont_labels, num_sex_labels = (
        x["y_true"].shape[0],
        len(binary_labels),
        len(continuous_label_names),
        demo_size,
    )

    ret_probs = {}
    ret_preds = {}

    probs = torch.zeros((batch_size, num_final_labels, num_cont_labels, num_sex_labels))

    bin_labels = [
        bin_label for bin_label in binary_labels if bin_label in ["slvh", "dlv"]
    ]

    for i, lab_name in enumerate(bin_labels):
        for cont_label in THRESHOLDS[lab_name]["M"].keys():
            idx = continuous_label_names.index(cont_label)
            probs[:, i, idx, m_index] = (
                1
                - x["normal_dist"].cdf(
                    torch.tensor(THRESHOLDS[lab_name]["M"][cont_label])
                )[:, idx]
            )
            probs[:, i, idx, f_index] = (
                1
                - x["normal_dist"].cdf(
                    torch.tensor(THRESHOLDS[lab_name]["F"][cont_label])
                )[:, idx]
            )

    preds = (probs >= 0.5).byte()

    sex_indeces = sex_info.nonzero()[:, -1]
    if "slvh" in binary_labels:
        ivs_idx = continuous_label_names.index("ivsd")
        lvw_idx = continuous_label_names.index("lvpwd")
        slvh_idx = bin_labels.index("slvh")

        slvh_probs = (
            probs[:, slvh_idx, ivs_idx, :]
            + probs[:, slvh_idx, lvw_idx, :]
            - probs[:, slvh_idx, [ivs_idx, lvw_idx], :].prod(axis=-2)
        )
        out_slvh_probs = slvh_probs[torch.arange(batch_size), sex_indeces]

        ret_probs["slvh"] = out_slvh_probs
        slvh_labs = preds[:, slvh_idx, ivs_idx, :] | preds[:, slvh_idx, lvw_idx, :]

        ret_preds["slvh"] = slvh_labs[torch.arange(batch_size), sex_indeces]

    if "dlv" in binary_labels:
        lvd_idx = continuous_label_names.index("lvidd")
        dcm_idx = bin_labels.index("dlv")
        # (batch_size, gender category)
        dcm_probs = probs[:, dcm_idx, lvd_idx, :]
        out_dcm_probs = dcm_probs[torch.arange(batch_size), sex_indeces]

        ret_probs["dlv"] = out_dcm_probs
        dcm_labs = preds[:, dcm_idx, lvd_idx, :]

        ret_preds["dlv"] = dcm_labs[torch.arange(batch_size), sex_indeces]

    if (set(["slvh", "dlv"]) - set(binary_labels)) == set():
        composite_slvh_dcm = torch.stack((out_slvh_probs, out_dcm_probs), dim=1)
        ret_probs["composite"] = 1.0 - (1.0 - composite_slvh_dcm).prod(axis=1)
        ret_preds["composite"] = ret_preds["slvh"] | ret_preds["dlv"]

    ret_probs = torch.hstack(
        [ret_probs[bin_lab].reshape(-1, 1) for bin_lab in binary_labels]
    )

    ret_preds = torch.hstack(
        [ret_preds[bin_lab].reshape(-1, 1) for bin_lab in binary_labels]
    )

    return ret_probs, ret_preds


def convert_continuous_to_binary(x, class_to_name, encoder):

    sex_info = x["demo_info"][:, :2]
    sex = encoder.inverse_transform(sex_info.cpu())
    categories_in_order = encoder.categories_[0].tolist()
    m_index, f_index = (
        categories_in_order.index("M"),
        categories_in_order.index("F"),
    )

    labels = ["slvh", "dlv"]

    # (batch_size, #final_label,s #cont_labels, gender category)
    batch_size, num_final_labels, num_cont_labels, num_sex_labels = (
        x["y_true"].shape[0],
        len(labels),
        x["y_true"].shape[1],
        2,
    )
    labs = torch.zeros(
        (batch_size, num_final_labels, num_cont_labels, num_sex_labels)
    ).byte()

    for i, lab_name in enumerate(labels):
        for cont_label in THRESHOLDS[lab_name]["M"].keys():
            idx = class_to_name.index(cont_label)
            labs[:, i, idx, m_index] = (
                x[:, idx] > THRESHOLDS[lab_name]["M"][cont_label]
            ).byte()
            labs[:, i, idx, f_index] = (
                x[:, idx] > THRESHOLDS[lab_name]["F"][cont_label]
            ).byte()
            labs[:, i, idx, o_index] = (
                x[:, idx] > THRESHOLDS[lab_name]["O"][cont_label]
            ).byte()

    ivs_idx = class_to_name.index("ivsd")
    lvw_idx = class_to_name.index("lvwpd")
    lvd_idx = class_to_name.index("lvidd")

    # (batch_size, gender category)
    slvh_labs = labs[:, 0, ivs_idx, :] | labs[:, 0, lvw_idx, :]
    dlv_labs = labs[:, 1, lvd_idx, :]

    sex_indeces = sex_info.nonzero()[:, -1]

    out_slvh_labs = slvh_labs[torch.arange(batch_size), sex_indeces]
    out_dlv_labs = dlv_labs[torch.arange(batch_size), sex_indeces]

    ret_labs = torch.stack((out_slvh_labs, out_dlv_labs), dim=1).float()

    return ret_labs.float()
