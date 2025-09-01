import torch
import torch.nn as nn

class Wasserstein_1d_Loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def forward(self, input, target, num_classes):
        # check if labels are same dimension with target
        assert input.shape[0] == target.shape[0], "Labels and target should have the same shape"
        with torch.no_grad():
            target_cdf = torch.nn.functional.one_hot(target, num_classes=num_classes)
            target_cdf = torch.cumsum(target_cdf, 1)
        input_cdf = torch.cumsum(input, 1)

        # doing reduction=mean here
        return torch.mean(torch.abs(input_cdf-target_cdf))

class Wasserstein_2_1d_Loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def forward(self, input, target, num_classes):
        # check if labels are same dimension with target
        assert input.shape[0] == target.shape[0], "Labels and target should have the same shape"
        with torch.no_grad():
            target_cdf = torch.nn.functional.one_hot(target, num_classes=num_classes)
            target_cdf = torch.cumsum(target_cdf, 1)
        input_cdf = torch.cumsum(input, 1)

        # doing reduction=mean here
        return torch.mean(torch.pow(input_cdf-target_cdf, 2))
