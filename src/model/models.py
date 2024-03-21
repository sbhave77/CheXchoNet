import torch
import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


class DenseNet121Base(nn.Module):
    """Simple base model
    Credit to arnoweng: https://github.com/arnoweng/CheXNet/blob/master/model.py
    """

    def __init__(self, out_size, model_kwargs={}, optuna_trial=None):
        super(DenseNet121Base, self).__init__()

        if "drop_rate" in model_kwargs and optuna_trial:
            drop_rate = optuna_trial.suggest_float("drop_rate", 0.3, 0.7, step=0.1)
            model_kwargs["drop_rate"] = drop_rate

        self.densenet121 = torchvision.models.densenet121(
            pretrained=False, **model_kwargs
        )
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_features, out_size),
        )

    def forward(self, x):
        return self.densenet121(x)


class DenseNet121BaseWithDemo(nn.Module):
    """Simple base model
    Credit to arnoweng: https://github.com/arnoweng/CheXNet/blob/master/model.py
    """

    def __init__(self, out_size, demo_size, model_kwargs={}, optuna_trial=None):
        super(DenseNet121BaseWithDemo, self).__init__()

        if "drop_rate" in model_kwargs and optuna_trial:
            drop_rate = optuna_trial.suggest_float("drop_rate", 0.3, 0.7, step=0.1)
            model_kwargs["drop_rate"] = drop_rate

        self.densenet121 = torchvision.models.densenet121(
            pretrained=False, **model_kwargs
        )
        num_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Identity()
        self.demo_mlp = nn.Sequential(
            nn.Linear(demo_size, 10),
            nn.ReLU(),
            nn.Linear(10, demo_size),
        )

        self.classifier = nn.Linear(demo_size + num_features, out_size)

    def forward(self, x, demo):
        demo_projected = self.demo_mlp(demo)
        out_pa = self.densenet121(x)
        pa_with_demo = torch.cat((demo_projected, out_pa), dim=1)
        out = self.classifier(pa_with_demo)

        return out


class DenseNet121MultiView(nn.Module):

    def __init__(self, out_size):
        super(DenseNet121MultiView, self).__init__()
        self.densenet121_PA = torchvision.models.densenet121(pretrained=False)
        self.densenet121_LL = torchvision.models.densenet121(pretrained=False)
        num_features = self.densenet121_PA.classifier.in_features
        self.densenet121_PA.classifier = nn.Identity()
        self.densenet121_LL.classifier = nn.Identity()

        self.classifier = nn.Linear(num_features * 2, out_size)

    def forward(self, x, y):

        out_pa = self.densenet121_PA(x)
        out_ll = self.densenet121_LL(y)

        pa_and_ll = torch.cat((out_pa, out_ll), dim=1)

        return self.classifier(pa_and_ll)


class EfficientNetDynamic(nn.Module):

    def __init__(self, out_size, input_size, pretrained=False):
        super(EfficientNetDynamic, self).__init__()
        input_size_to_model_name = {
            224: "efficientnet-b0",
            240: "efficientnet-b1",
            260: "efficientnet-b2",
            300: "efficientnet-b3",
            380: "efficientnet-b4",
            456: "efficientnet-b5",
            528: "efficientnet-b6",
            600: "efficientnet-b7",
            672: "efficientnet-b8",
            800: "efficientnet-b9",
        }
        if pretrained:
            self.efficient_net = EfficientNet.from_pretrained(
                input_size_to_model_name[input_size]
            )
        else:
            self.efficient_net = EfficientNet.from_pretrained(
                input_size_to_model_name[input_size]
            )

        self.efficient_net._fc = nn.Linear(self.efficient_net._fc.in_features, out_size)

    def forward(self, x):
        return self.efficient_net(x)
