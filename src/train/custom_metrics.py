from typing import Any, Callable, Tuple, Union
import torch
from ignite.metrics import EpochMetric
from sklearn.metrics import confusion_matrix

def specificity_compute_fn(y_preds: torch.Tensor, y_targets: torch.Tensor) -> float:
    y_true = y_targets.cpu().numpy()
    y_pred = y_preds.cpu().numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn+fp)
    sens = tp / (tp+fn)
    
    return 1 - spec, sens



class SpecAndSens(EpochMetric):
    def __init__(
            self,
            output_transform: Callable = lambda x: x,
            check_compute_fn: bool = False,
            device: Union[str, torch.device] = torch.device("cpu"),
    ):
        
        super(SpecAndSens, self).__init__(
            specificity_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn, device=device
        )
