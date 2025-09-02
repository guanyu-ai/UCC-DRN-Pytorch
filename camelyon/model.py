import sys
sys.path.append("../")
from drn_pytorch.drn import DRN
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

# from drn import DRNLayer
torch.autograd.set_detect_anomaly(True)


class ResBlockZeroPadding(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, upsample=False, first_block=False):
        super().__init__()
        assert not (
            upsample and downsample), "Only set upsample or downsample as true"
        self.upsample = upsample
        self.downsample = downsample
        self.first_block = first_block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
            stride=2 if self.downsample else 1
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1
        )
        self.skip_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=2 if self.downsample else 1
        )
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0.1)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0.1)
        nn.init.xavier_uniform_(self.skip_conv.weight)

        if self.upsample:
            self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, input_tensor):
        if self.first_block:
            input_tensor = F.relu(input_tensor)
            if self.upsample:
                input_tensor = self.upsampling(input_tensor)

        x = F.relu(self.conv1(input_tensor))
        x = F.relu(self.conv2(x))
        if input_tensor.shape != x.shape:   # check if the number of channels are correct.
            input_tensor = self.skip_conv(input_tensor)
        output = x + input_tensor
        return output


class WideResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, downsample=False, upsample=False):
        super(WideResidualBlock, self).__init__()
        self.blocks = nn.Sequential(
            *[
                ResBlockZeroPadding(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    first_block=(i == 0),
                    downsample=downsample,
                    upsample=upsample
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, x):
        return self.blocks(x)


class Reshape(nn.Module):
    def __init__(self, *target_shape):
        super(Reshape, self).__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(x.size(0), *self.target_shape)

class UCCModel(nn.Module):
    def __init__(self, cfg):
        super(UCCModel, self).__init__()
        model_cfg = cfg.model
        args = cfg.args
        # self.g_softmax = GumbelSoftmax()
        self.num_instances = args.num_instances
        self.num_features = model_cfg.encoder.num_features
        self.num_channels = model_cfg.num_channels
        self.num_classes = 2
        self.batch_size = args.batch_size
        self.alpha = model_cfg.loss.alpha
        self.sigma = model_cfg.kde_model.sigma
        self.num_nodes = model_cfg.kde_model.num_bins
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=cfg.model.encoder.conv_input_channel,
                out_channels=cfg.model.encoder.conv_output_channel,
                kernel_size=(3, 3),
                padding=1),
            WideResidualBlock(
                in_channels=cfg.model.encoder.conv_output_channel,
                out_channels=cfg.model.encoder.block1_output_channel,
                n_layers=cfg.model.encoder.block1_num_layer
            ),
            WideResidualBlock(
                in_channels=cfg.model.encoder.block1_output_channel,
                out_channels=cfg.model.encoder.block2_output_channel,
                n_layers=cfg.model.encoder.block2_num_layer,
                downsample=True
            ),
            WideResidualBlock(
                in_channels=cfg.model.encoder.block2_output_channel,
                out_channels=cfg.model.encoder.block3_output_channel,
                n_layers=cfg.model.encoder.block3_num_layer,
                downsample=True
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=cfg.model.encoder.flatten_size,
                      out_features=self.num_features, bias=False),
            nn.Sigmoid()
        )
        if self.alpha == 1:
            self.decoder = None
        else:
            self.decoder = nn.Sequential(
                nn.Linear(in_features=self.num_features,
                          out_features=cfg.model.decoder.linear_size),
                nn.ReLU(),
                nn.Unflatten(-1, [128, 8, 8]),
                WideResidualBlock(
                    in_channels=128, out_channels=64, n_layers=1, upsample=True),
                WideResidualBlock(
                    in_channels=64, out_channels=32, n_layers=1, upsample=True),
                WideResidualBlock(
                    in_channels=32, out_channels=16, n_layers=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=self.num_channels,
                          kernel_size=(3, 3), padding=1)
            )
        self.ucc_classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(in_features=self.num_features*self.num_nodes, out_features=384),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=384, out_features=192),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(in_features=192, out_features=self.num_classes),
            nn.Softmax(dim=-1)
        )

    def kde(self, data: torch.Tensor, num_nodes, sigma):
        device = data.device
        # data shape: (batch_size, num_instances, num_features)
        batch_size, num_instances, num_features = data.shape

        # Create sample points
        k_sample_points = (
            torch.linspace(0, 1, steps=num_nodes)
            .repeat(batch_size, num_instances, 1)
            .to(device)
        )

        # Calculate constants
        k_alpha = 1 / np.sqrt(2 * np.pi * sigma**2)
        k_beta = -1 / (2 * sigma**2)

        # Iterate over features and calculate kernel density estimation for each feature
        out_list = []
        for i in range(num_features):
            one_feature = data[:, :, i: i + 1].repeat(1, 1, num_nodes)
            k_diff_2 = (k_sample_points - one_feature) ** 2
            k_result = k_alpha * torch.exp(k_beta * k_diff_2)
            k_out_unnormalized = k_result.sum(dim=1)
            k_norm_coeff = k_out_unnormalized.sum(dim=1).view(-1, 1)
            k_out = k_out_unnormalized / k_norm_coeff.repeat(
                1, k_out_unnormalized.size(1)
            )
            out_list.append(k_out)

        # Concatenate the results
        concat_out = torch.cat(out_list, dim=-1)
        return concat_out

    def forward(self, x, return_reconstruction=False):
        # x shape: (batch_size, num_instances, num_channel, patch_size, patch_size)
        batch_size, num_instances, num_channel, patch_size, _ = x.shape
        # reshape x to (batch_size * num_instances, 1, patch_size, patch_size)
        x = x.view(-1, num_channel, x.shape[-2], x.shape[-1])
        feature = self.encoder(x)
        # if self.decoder:
        #     reconstruction = self.decoder(feature)
        #     res = reconstruction.view(batch_size, num_instances, 1, patch_size, patch_size)
        # feature = self.patch_model(x)
        # reshape output to (batch_size, num_instances, num_features)
        feature_ = feature.view(batch_size, num_instances, feature.shape[-1])
        # use kernel density estimation to estimate the distribution of the features
        # output of kde is concatenated features distribution
        feature_distribution = self.kde(feature_, self.num_nodes, self.sigma)

        out = self.ucc_classifier(feature_distribution)
        if return_reconstruction:
            reconstruction = self.decoder(feature)
            res = reconstruction.view(
                batch_size, num_instances, num_channel, patch_size, patch_size)
            return out, res
        return out

    def compute_loss(self, inputs=None, labels=None, output=None, reconstruction=None, return_losses=False):
        # if labels is not None:
        if self.alpha == 1:
            assert not isinstance(output, type(
                None)), "Output classes must be provided"
            return F.cross_entropy(output, labels)
        elif self.alpha == 0:
            assert not isinstance(reconstruction, type(
                None)), "Reconstruction must be provided"
            return F.mse_loss(reconstruction, inputs)
        else:
            assert not isinstance(output, type(
                None)), "Output classes must be provided"
            assert not isinstance(labels, type(
                None)), "Labels must be provided"
            assert not isinstance(inputs, type(
                None)), "Inputs must be provided"
            assert not isinstance(reconstruction, type(
                None)), "Reconstructed input must be provided"
            if return_losses:
                ce_loss = self.alpha*F.cross_entropy(output, labels)
                rec_loss = (1-self.alpha)*F.mse_loss(reconstruction, inputs)
                return ce_loss, rec_loss, ce_loss+rec_loss
            else:
                return self.alpha*F.cross_entropy(output, labels) + (1-self.alpha)*F.mse_loss(reconstruction, inputs)

    def reconstruct_image(self, features):
        assert self.decoder is not None, "Model does not have a decoder"
        return self.decoder(features)

    def ucc_model(self, inputs):
        feature_list = list()
        for i in range(self.batch_size):
            temp_input = inputs[i, :, :, :, :]
            temp_feature = self.encoder(temp_input)
            temp_feature = torch.reshape(
                temp_feature, (1, self.num_instances, self.num_features))
            feature_list.append(temp_feature)

        feature_concatenated = torch.concatenate(feature_list, dim=0)
        feature_distributions = self.kde(
            feature_concatenated, self.num_nodes, self.sigma)
        output = self.ucc_classifier(feature_distributions)
        return output

    def extract_features(self, x):
        # x shape: (batch_size, num_instances, num_channel, patch_size, patch_size)
        batch_size, num_instances, num_channel, patch_size, _ = x.shape
        # reshape x to (batch_size * num_instances, 1, patch_size, patch_size)
        x = x.view(-1, num_channel, x.shape[-2], x.shape[-1])
        feature = self.encoder(x)
        # if self.decoder:
        #     reconstruction = self.decoder(feature)
        #     res = reconstruction.view(batch_size, num_instances, 1, patch_size, patch_size)
        # feature = self.patch_model(x)
        # reshape output to (batch_size, num_instances, num_features)
        feature_ = feature.view(batch_size, num_instances, feature.shape[-1])
        # use kernel density estimation to estimate the distribution of the features
        # output of kde is concatenated features distribution
        feature_distribution = self.kde(feature_, self.num_nodes, self.sigma)
        return feature_distribution


class UCCDRNModel(UCCModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        kwargs = {}
        if "init_method" in cfg.model.drn:
            self.init_method = cfg.model.drn.init_method
            if self.init_method == "normal":
                kwargs["init_method"] = "normal"
                kwargs["mean"] = cfg.model.drn.init_mean
                kwargs["std"] = cfg.model.drn.init_std
            elif self.init_method == "uniform":
                kwargs["init_method"] = "uniform"
                kwargs["upper_bound"] = cfg.model.drn.init_upper_bound
                kwargs["lower_bound"] = cfg.model.drn.init_lower_bound
        self.ucc_classifier = nn.Sequential(
            DRN(cfg.model.encoder.num_features,
                cfg.model.drn.num_bins,
                cfg.model.drn.num_nodes,
                cfg.model.drn.hidden_q,
                **kwargs
                ),
            *[DRN(
                cfg.model.drn.num_nodes,
                cfg.model.drn.hidden_q,
                cfg.model.drn.num_nodes,
                cfg.model.drn.hidden_q,
                **kwargs
            )
                for i in range(cfg.model.drn.num_layers-1)],
            DRN(
                cfg.model.drn.num_nodes,
                cfg.model.drn.hidden_q,
                1,
                cfg.model.drn.output_bins,
                **kwargs
            ),
            nn.Flatten()
        )


class DRNOnlyModel(nn.Module):
    def __init__(self, cfg):
        super(DRNOnlyModel, self).__init__()
        self.drn = nn.Sequential(
            DRN(cfg.model.encoder.num_features,
                cfg.model.drn.num_bins,
                cfg.model.drn.num_nodes,
                cfg.model.drn.hidden_q
                ),
            *[DRN(
                cfg.model.drn.num_nodes,
                cfg.model.drn.hidden_q,
                cfg.model.drn.num_nodes,
                cfg.model.drn.hidden_q
            )
                for i in range(cfg.model.drn.num_layers-1)],
            DRN(
                cfg.model.drn.num_nodes,
                cfg.model.drn.hidden_q,
                1,
                cfg.model.drn.output_bins
            ),
            nn.Flatten()
        )
        self.num_bins = cfg.model.kde_model.num_bins
        self.sigma = cfg.model.kde_model.sigma
        self.num_features = cfg.model.encoder.num_features

    def kde(self, data: torch.Tensor, num_nodes, sigma):
        device = data.device
        # data shape: (batch_size, num_instances, num_features)
        batch_size, num_instances, num_features = data.shape

        # Create sample points
        k_sample_points = (
            torch.linspace(0, 1, steps=num_nodes)
            .repeat(batch_size, num_instances, 1)
            .to(device)
        )

        # Calculate constants
        k_alpha = 1 / np.sqrt(2 * np.pi * sigma**2)
        k_beta = -1 / (2 * sigma**2)

        # Iterate over features and calculate kernel density estimation for each feature
        out_list = []
        for i in range(num_features):
            one_feature = data[:, :, i: i + 1].repeat(1, 1, num_nodes)
            k_diff_2 = (k_sample_points - one_feature) ** 2
            k_result = k_alpha * torch.exp(k_beta * k_diff_2)
            k_out_unnormalized = k_result.sum(dim=1)
            k_norm_coeff = k_out_unnormalized.sum(dim=1).view(-1, 1)
            k_out = k_out_unnormalized / k_norm_coeff.repeat(
                1, k_out_unnormalized.size(1)
            )
            out_list.append(k_out)

        # Concatenate the results
        concat_out = torch.cat(out_list, dim=-1)
        return concat_out

    def forward(self, x):
        x = self.kde(x, self.num_bins, self.sigma)
        x = x.reshape(-1, self.num_features, self.num_bins)
        output = self.drn(x)
        return output

    def compute_loss(self, labels, outputs):
        return F.cross_entropy(outputs, labels)
