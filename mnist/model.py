# import tensorflow as tf
import numpy as np

import os
import sys

sys.path.append("../")

# from drn import DRNLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from drn_pytorch.drn import DRN
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
        self.num_classes = args.ucc_end-args.ucc_start+1
        self.batch_size = args.num_samples_per_class*self.num_classes
        self.alpha = model_cfg.loss.alpha
        self.sigma = model_cfg.kde_model.sigma
        self.num_nodes = model_cfg.kde_model.num_bins
        self.encoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=(3, 3),
                    padding=1),
                WideResidualBlock(
                    in_channels=16,
                    out_channels=32,
                    n_layers=1
                ),
                WideResidualBlock(
                    in_channels=32,
                    out_channels=64,
                    n_layers=1,
                    downsample=True
                ),
                WideResidualBlock(
                    in_channels=64,
                    out_channels=128,
                    n_layers=1,
                    downsample=True
                ),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(in_features=6272, out_features=self.num_features, bias=False),
                nn.Sigmoid()
        )
        if self.alpha == 1:
            self.decoder = None
        else:
            self.decoder = nn.Sequential(
                    nn.Linear(in_features=self.num_features,
                              out_features=7*7*128),
                    nn.ReLU(),
                    nn.Unflatten(-1, [128, 7, 7]),
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
                nn.Linear(in_features=110, out_features=384),
                nn.ReLU(),
                nn.Linear(in_features=384, out_features=192),
                nn.ReLU(),
                nn.Linear(in_features=192, out_features=self.num_classes),
                nn.Softmax(dim=-1)
        )

    def kde(self, data:torch.Tensor, num_nodes, sigma):
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
            one_feature = data[:, :, i : i + 1].repeat(1, 1, num_nodes)
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

    def forward(self, x, label=None):
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
        if self.decoder:
            reconstruction = self.decoder(feature)
            res = reconstruction.view(batch_size, num_instances, 1, patch_size, patch_size)
            return out, res
        return out

    def compute_loss(self, inputs=None, labels=None, output=None, reconstruction=None):
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
            return self.alpha*F.cross_entropy(output, labels)+ (1-self.alpha)*F.mse_loss(reconstruction, inputs)

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

    def extract_features(self, inputs):
        feature_list = list()
        for i in range(self.batch_size):
            temp_input = inputs[i, :, :, :, :]
            temp_feature = self.encoder(temp_input)
            temp_feature = torch.reshape(
                temp_feature, (1, self.num_instances, self.num_features))
            feature_list.append(temp_feature)

        feature_concatenated = torch.concatenate(feature_list, dim=0)
        return feature_concatenated


class UCCDRNModel(UCCModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ucc_classifier = nn.Sequential(
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


# class UCCDRNModel(tf.keras.Model):
#     def __init__(self, cfg):
#         super(UCCDRNModel, self).__init__()
#         model_cfg = cfg.model
#         args = cfg.args
#         self.num_instances = args.num_instances
#         self.num_features = model_cfg.encoder.num_features
#         self.num_channels = model_cfg.num_channels
#         self.num_classes = args.ucc_end-args.ucc_start+1
#         self.batch_size = args.num_samples_per_class*self.num_classes

#         self.num_bins = model_cfg.kde_model.num_bins
#         self.sigma = model_cfg.kde_model.sigma

#         self.drn = None
#         self.drn_num_bins = model_cfg.drn.num_bins
#         self.drn_hidden_q = model_cfg.drn.hidden_q
#         self.drn_num_nodes = model_cfg.drn.num_nodes
#         self.drn_num_layers = model_cfg.drn.num_layers
#         self.drn_output_bins = model_cfg.drn.output_bins
#         self.gumbel_tau = model_cfg.drn.gumbel.tau

#         self.alpha = model_cfg.loss.alpha

#         self.encoder = tf.keras.Sequential(
#             [
#                 tf.keras.Input((28, 28, 1)),
#                 tf.keras.layers.Conv2D(16, (3, 3), padding='same'),
#                 WideResidualBlock(filters=32, n_layers=1),
#                 WideResidualBlock(filters=64, n_layers=1,  downsample=True),
#                 WideResidualBlock(filters=128, n_layers=1,  downsample=True),
#                 tf.keras.layers.ReLU(),
#                 tf.keras.layers.Reshape((6272,)),
#                 tf.keras.layers.Dense(self.num_features, activation='sigmoid')

#             ],
#             name="encoder"
#         )
#         if self.alpha == 1:
#             self.decoder = None
#         else:
#             self.decoder = tf.keras.Sequential(
#                 [
#                     tf.keras.Input((self.num_features,)),
#                     tf.keras.layers.Dense(7*7*128, activation='relu'),
#                     tf.keras.layers.Reshape([7, 7, 128]),
#                     WideResidualBlock(filters=64, n_layers=1, upsample=True, ),
#                     WideResidualBlock(filters=32, n_layers=1, upsample=True, ),
#                     WideResidualBlock(filters=16, n_layers=1, ),
#                     tf.keras.layers.ReLU(),
#                     tf.keras.layers.Conv2D(
#                         filters=self.num_channels, kernel_size=(3, 3), padding='same')
#                 ],
#                 name="decoder"
#             )
#         if model_cfg.drn:
#             drn_num_layers = model_cfg.drn.num_layers
#             layers = []
#             for i in range(drn_num_layers+1):
#                 layers.append(
#                     DRNLayer(
#                         n_lower=self.num_features if i == 0 else self.drn_num_nodes,
#                         n_upper=1 if i == drn_num_layers else self.drn_num_nodes,
#                         q_lower=self.drn_num_bins if i == 0 else self.drn_hidden_q,
#                         q_upper=self.drn_output_bins if i == drn_num_layers else self.drn_hidden_q,
#                     )
#                 )
#             self.drn = tf.keras.Sequential(
#                 [
#                     tf.keras.Input((self.num_features, self.num_bins)),
#                     *layers,
#                     tf.keras.layers.Reshape([self.drn_output_bins,])
#                 ],
#                 name="drn"
#             )
#             self.gumbel = GumbelSoftmax()

#         if model_cfg.ucc_classifier != "None":

#             self.ucc_classifier = tf.keras.Sequential(
#                 [
#                     tf.keras.Input((self.drn_output_bins, )),
#                     tf.keras.layers.Dense(4, activation='softmax'),
#                 ],
#                 name="ucc_classifier"
#             )
#             print("ucc classifier initialize ")
#         else:
#             self.ucc_classifier = None

#     def kde(self, data, num_nodes, sigma):
#         # our input data shape is [batch_size, num_instances, num_features]
#         # batchsize x bins
#         # we want each linspace to represent a distribution for each feature of each instance. Expected shape = [batch_size, num_instance, num_nodes]
#         k_sample_points = tf.constant(
#             np.tile(np.linspace(0, 1, num=num_nodes), [self.num_instances, 1]).astype(np.float32))
#         k_alpha = tf.constant(
#             np.array(1/np.sqrt(2*np.pi*np.square(sigma))).astype(np.float32))
#         k_beta = tf.constant(
#             np.array(-1/(2*np.square(sigma))).astype(np.float32))
#         out = []
#         # for concatenating across each feature point
#         for i in range(self.num_features):
#             # For each feature point
#             # Reshape for broadcasting
#             temp = tf.reshape(
#                 data[:, :, i], (self.batch_size, self.num_instances, 1))
#             # get x-x_0 values into a grid
#             k_diff = k_sample_points - tf.tile(temp, [1, 1, num_nodes])
#             diff_sq = tf.square(k_diff)
#             k_result = k_alpha * tf.exp(k_beta*diff_sq)
#             # add all the feature values across instances. Expected shape = [batch_size, num_nodes]
#             k_out_unnormalized = tf.reduce_sum(k_result, axis=1)
#             k_norm_coeff = tf.reshape(tf.reduce_sum(
#                 k_out_unnormalized, axis=1), (-1, 1))
#             k_out = k_out_unnormalized / \
#                 tf.tile(k_norm_coeff, [1, k_out_unnormalized.shape[1]])
#             out.append(tf.reshape(k_out, [self.batch_size, 1, num_nodes]))
#         # Expected output shape =[batch_size,num_features, num_nodes]
#         concat_out = tf.concat(out, axis=1)
#         return concat_out

#     def call(self, inputs):
#         input_list = list()
#         feature_list = list()
#         ae_output_list = list()
#         if self.decoder:
#             for i in range(self.batch_size):
#                 temp_input = inputs[i, :, :, :, :]
#                 temp_feature = self.encoder(temp_input)
#                 temp_feature = tf.reshape(
#                     temp_feature, (1, self.num_instances, self.num_features))

#                 temp_ae_output = self.decoder(self.encoder(temp_input))
#                 temp_ae_output = tf.reshape(
#                     temp_ae_output, (1, self.num_instances, 28, 28, self.num_channels))
#                 input_list.append(temp_input)
#                 feature_list.append(temp_feature)
#                 ae_output_list.append(temp_ae_output)
#             feature_concatenated = tf.concat(feature_list, axis=0)
#             feature_distributions = self.kde(
#                 feature_concatenated, self.num_bins, self.sigma)
#             drn_output = self.drn(feature_distributions)
#             output = self.ucc_classifier(
#                 drn_output) if self.ucc_classifier else drn_output
#             # output = GumbelSoftmax()(output, tau=self.gumbel_tau)
#             ae_output = tf.concat(ae_output_list, axis=0)
#             # return output[0], ae_output
#             return output, ae_output
#         else:
#             for i in range(self.batch_size):
#                 temp_input = inputs[i, :, :, :, :]
#                 temp_feature = self.encoder(temp_input)
#                 temp_feature = tf.reshape(
#                     temp_feature, (1, self.num_instances, self.num_features))

#                 input_list.append(temp_input)
#                 feature_list.append(temp_feature)
#             feature_concatenated = tf.concat(feature_list, axis=0)
#             feature_distributions = self.kde(
#                 feature_concatenated, self.num_bins, self.sigma)
#             drn_output = self.drn(feature_distributions)
#             output = self.ucc_classifier(
#                 drn_output) if self.ucc_classifier else drn_output
#             # output = GumbelSoftmax()(output, tau=self.gumbel_tau)
#             # return output[0]
#             return output

#     def compute_loss(self, inputs=None, labels=None, outputs=None, reconstruction=None):
#         if self.alpha == 1:
#             assert not isinstance(outputs, type(
#                 None)), "Output classes must be provided"
#             cce = tf.keras.losses.CategoricalCrossentropy()
#             return cce(labels, outputs)
#         elif self.alpha == 0:
#             mse = tf.keras.losses.MeanSquaredError()
#             return mse(reconstruction, inputs)
#         else:
#             if reconstruction != None:
#                 cce = tf.keras.losses.CategoricalCrossentropy()
#                 mse = tf.keras.losses.MeanSquaredError()
#                 return cce(labels, outputs), mse(reconstruction, inputs)
#             else:
#                 cce = tf.keras.losses.CategoricalCrossentropy()
#                 return cce(labels, outputs)

#     def reconstruct_image(self, features):
#         assert self.decoder != None, "Model does not have a decoder"
#         return self.decoder(features)

#     def get_feature_distribution(self, inputs):
#         feature_list = list()
#         for i in range(inputs.shape[0]):
#             temp_input = tf.expand_dims(inputs[i, :, :, :], axis=0)
#             temp_feature = self.encoder(temp_input)
#             temp_feature = tf.reshape(
#                 temp_feature, (1, self.num_instances, self.num_features))
#             feature_list.append(temp_feature)
#         feature_concatenated = tf.concat(feature_list, axis=0)
#         return self.kde(feature_concatenated, self.num_nodes, self.sigma)

#     def ucc_classify(self, inputs):
#         input_list = list()
#         feature_list = list()
#         for i in range(self.batch_size):
#             temp_input = inputs[i, :, :, :, :]
#             temp_feature = self.encoder(temp_input)
#             temp_feature = tf.reshape(
#                 temp_feature, (1, self.num_instances, self.num_features))

#             input_list.append(temp_input)
#             feature_list.append(temp_feature)
#         feature_concatenated = tf.concat(feature_list, axis=0)
#         feature_distributions = self.kde(
#             feature_concatenated, self.num_bins, self.sigma)
#         drn_output = self.drn(feature_distributions)
#         output = self.ucc_classifier(
#             drn_output) if self.ucc_classifier else drn_output
#         # output = GumbelSoftmax()(output, tau=self.gumbel_tau)
#         # return output[0]
#         return output

#     def autoencode(self, inputs):
#         # not sure if we need this or not. Could be useful but not needed atm.
#         pass

#     def extract_features(self, inputs):
#         steps = self.batch_size*self.num_instances
#         length = inputs.shape[0]
#         remainder = length % steps
#         feature_list = []
#         for i in range(np.ceil(length/steps).astype(np.int32)):
#             start = i*steps
#             if i+1 > np.ceil(length/steps):
#                 stop = start + remainder
#             else:
#                 stop = (i+1)*steps
#             input_array = inputs[start:stop, :, :, :]
#             feature_list.append(self.encoder(input_array))
#         feature_concatenated = tf.concat(feature_list, axis=0)
#         return feature_concatenated


# class DRNOnlyModel(tf.keras.Model):
#     def __init__(self, cfg):
#         super(DRNOnlyModel, self).__init__()
#         self.drn = None
#         self.num_instances = cfg.args.num_instances
#         self.num_features = cfg.args.num_features
#         self.num_classes = cfg.args.ucc_end-cfg.args.ucc_start+1
#         self.batch_size = cfg.args.num_samples_per_class*self.num_classes
#         self.sigma = cfg.model.kde_model.sigma
#         self.num_bins = cfg.model.kde_model.num_bins
#         self.input_bins = self.num_bins
#         self.hidden_q = cfg.model.drn.hidden_q
#         self.num_nodes = cfg.model.drn.num_nodes
#         self.num_layers = cfg.model.drn.num_layers
#         self.output_bins = cfg.model.drn.output_bins
#         if cfg.model.drn:
#             print('initializing drn model')
#             layers = []
#             for i in range(self.num_layers+1):
#                 layers.append(
#                     DRNLayer(
#                         n_lower=self.num_features if i == 0 else self.num_nodes,
#                         n_upper=1 if i == self.num_layers else self.num_nodes,
#                         q_lower=self.num_bins if i == 0 else self.hidden_q,
#                         q_upper=self.output_bins if i == self.num_layers else self.hidden_q,
#                     )
#                 )
#             self.drn = tf.keras.Sequential(
#                 [
#                     tf.keras.Input((self.num_features, self.num_bins)),
#                     *layers,
#                     tf.keras.layers.Reshape([self.output_bins,])
#                 ],
#                 name="drn"
#             )

#     def kde(self, data, num_nodes, sigma):
#         # our input data shape is [batch_size, num_instances, num_features]
#         # batchsize x bins
#         # we want each linspace to represent a distribution for each feature of each instance. Expected shape = [batch_size, num_instance, num_nodes]
#         k_sample_points = tf.constant(
#             np.tile(np.linspace(0, 1, num=num_nodes), [self.num_instances, 1]).astype(np.float32))
#         k_alpha = tf.constant(
#             np.array(1/np.sqrt(2*np.pi*np.square(sigma))).astype(np.float32))
#         k_beta = tf.constant(
#             np.array(-1/(2*np.square(sigma))).astype(np.float32))
#         out = []
#         # for concatenating across each feature point
#         for i in range(self.num_features):
#             # For each feature point
#             # Reshape for broadcasting
#             temp = tf.reshape(
#                 data[:, :, i], (self.batch_size, self.num_instances, 1))
#             # get x-x_0 values into a grid
#             k_diff = k_sample_points - tf.tile(temp, [1, 1, num_nodes])
#             diff_sq = tf.square(k_diff)
#             k_result = k_alpha * tf.exp(k_beta*diff_sq)
#             # add all the feature values across instances. Expected shape = [batch_size, num_nodes]
#             k_out_unnormalized = tf.reduce_sum(k_result, axis=1)
#             k_norm_coeff = tf.reshape(tf.reduce_sum(
#                 k_out_unnormalized, axis=1), (-1, 1))
#             k_out = k_out_unnormalized / \
#                 tf.tile(k_norm_coeff, [1, k_out_unnormalized.shape[1]])
#             out.append(tf.reshape(k_out, [self.batch_size, 1, num_nodes]))
#         # Expected output shape =[batch_size,num_features, num_nodes]
#         concat_out = tf.concat(out, axis=1)
#         return concat_out

#     def call(self, inputs):
#         x = self.kde(inputs, self.num_bins, self.sigma)
#         output = self.drn(x)
#         return output

#     def compute_loss(self, labels, outputs):
#         cce = tf.keras.losses.CategoricalCrossentropy()
#         return cce(labels, outputs)


# class UCCDRNModelUnnormal(tf.keras.Model):
#     def __init__(self, cfg):
#         super(UCCDRNModelUnnormal, self).__init__()
#         model_cfg = cfg.model
#         args = cfg.args
#         self.num_instances = args.num_instances
#         self.num_features = model_cfg.encoder.num_features
#         self.num_channels = model_cfg.num_channels
#         self.num_classes = args.ucc_end-args.ucc_start+1
#         self.batch_size = args.num_samples_per_class*self.num_classes

#         self.num_bins = model_cfg.kde_model.num_bins
#         self.sigma = model_cfg.kde_model.sigma

#         self.drn = None
#         self.drn_num_bins = model_cfg.drn.num_bins
#         self.drn_hidden_q = model_cfg.drn.hidden_q
#         self.drn_num_nodes = model_cfg.drn.num_nodes
#         self.drn_num_layers = model_cfg.drn.num_layers
#         self.drn_output_bins = model_cfg.drn.output_bins
#         self.gumbel_tau = model_cfg.drn.gumbel.tau

#         self.alpha = model_cfg.loss.alpha

#         self.encoder = tf.keras.Sequential(
#             [
#                 tf.keras.Input((28, 28, 1)),
#                 tf.keras.layers.Conv2D(16, (3, 3), padding='same'),
#                 WideResidualBlock(filters=32, n_layers=1),
#                 WideResidualBlock(filters=64, n_layers=1,  downsample=True),
#                 WideResidualBlock(filters=128, n_layers=1,  downsample=True),
#                 tf.keras.layers.ReLU(),
#                 tf.keras.layers.Reshape((6272,)),
#                 tf.keras.layers.Dense(self.num_features, activation='sigmoid')

#             ],
#             name="encoder"
#         )
#         if self.alpha == 1:
#             self.decoder = None
#         else:
#             self.decoder = tf.keras.Sequential(
#                 [
#                     tf.keras.Input((self.num_features,)),
#                     tf.keras.layers.Dense(7*7*128, activation='relu'),
#                     tf.keras.layers.Reshape([7, 7, 128]),
#                     WideResidualBlock(filters=64, n_layers=1, upsample=True, ),
#                     WideResidualBlock(filters=32, n_layers=1, upsample=True, ),
#                     WideResidualBlock(filters=16, n_layers=1, ),
#                     tf.keras.layers.ReLU(),
#                     tf.keras.layers.Conv2D(
#                         filters=self.num_channels, kernel_size=(3, 3), padding='same')
#                 ],
#                 name="decoder"
#             )
#         if model_cfg.drn:
#             drn_num_layers = model_cfg.drn.num_layers
#             layers = []
#             for i in range(drn_num_layers+1):
#                 layers.append(
#                     DRNLayer(
#                         n_lower=self.num_features if i == 0 else self.drn_num_nodes,
#                         n_upper=1 if i == drn_num_layers else self.drn_num_nodes,
#                         q_lower=self.drn_num_bins if i == 0 else self.drn_hidden_q,
#                         q_upper=self.drn_output_bins if i == drn_num_layers else self.drn_hidden_q,
#                         normalize=(i != 0)
#                     )
#                 )
#             self.drn = tf.keras.Sequential(
#                 [
#                     tf.keras.Input((self.num_features, self.num_bins)),
#                     *layers,
#                     tf.keras.layers.Reshape([self.drn_output_bins,])
#                 ],
#                 name="drn"
#             )
#             self.gumbel = GumbelSoftmax()

#         if model_cfg.ucc_classifier != "None":

#             self.ucc_classifier = tf.keras.Sequential(
#                 [
#                     tf.keras.Input((self.drn_output_bins, )),
#                     tf.keras.layers.Dense(4, activation='softmax'),
#                 ],
#                 name="ucc_classifier"
#             )
#             print("ucc classifier initialize ")
#         else:
#             self.ucc_classifier = None

#     def kde(self, data, num_nodes, sigma):
#         # our input data shape is [batch_size, num_instances, num_features]
#         # batchsize x bins
#         # we want each linspace to represent a distribution for each feature of each instance. Expected shape = [batch_size, num_instance, num_nodes]
#         k_sample_points = tf.constant(
#             np.tile(np.linspace(0, 1, num=num_nodes), [self.num_instances, 1]).astype(np.float32))
#         k_alpha = tf.constant(
#             np.array(1/np.sqrt(2*np.pi*np.square(sigma))).astype(np.float32))
#         k_beta = tf.constant(
#             np.array(-1/(2*np.square(sigma))).astype(np.float32))
#         out = []
#         # for concatenating across each feature point
#         for i in range(self.num_features):
#             # For each feature point
#             # Reshape for broadcasting
#             temp = tf.reshape(
#                 data[:, :, i], (self.batch_size, self.num_instances, 1))
#             # get x-x_0 values into a grid
#             k_diff = k_sample_points - tf.tile(temp, [1, 1, num_nodes])
#             diff_sq = tf.square(k_diff)
#             k_result = k_alpha * tf.exp(k_beta*diff_sq)
#             # add all the feature values across instances. Expected shape = [batch_size, num_nodes]
#             k_out_unnormalized = tf.reduce_sum(k_result, axis=1)
#             k_norm_coeff = tf.reshape(tf.reduce_sum(
#                 k_out_unnormalized, axis=1), (-1, 1))
#             k_out = k_out_unnormalized / \
#                 tf.tile(k_norm_coeff, [1, k_out_unnormalized.shape[1]])
#             out.append(tf.reshape(k_out, [self.batch_size, 1, num_nodes]))
#         # Expected output shape =[batch_size,num_features, num_nodes]
#         concat_out = tf.concat(out, axis=1)
#         return concat_out

#     def call(self, inputs):
#         input_list = list()
#         feature_list = list()
#         ae_output_list = list()
#         if self.decoder:
#             for i in range(self.batch_size):
#                 temp_input = inputs[i, :, :, :, :]
#                 temp_feature = self.encoder(temp_input)
#                 temp_feature = tf.reshape(
#                     temp_feature, (1, self.num_instances, self.num_features))

#                 temp_ae_output = self.decoder(self.encoder(temp_input))
#                 temp_ae_output = tf.reshape(
#                     temp_ae_output, (1, self.num_instances, 28, 28, self.num_channels))
#                 input_list.append(temp_input)
#                 feature_list.append(temp_feature)
#                 ae_output_list.append(temp_ae_output)
#             feature_concatenated = tf.concat(feature_list, axis=0)
#             feature_distributions = self.kde(
#                 feature_concatenated, self.num_bins, self.sigma)
#             drn_output = self.drn(feature_distributions)
#             output = self.ucc_classifier(
#                 drn_output) if self.ucc_classifier else drn_output
#             # output = GumbelSoftmax()(output, tau=self.gumbel_tau)
#             ae_output = tf.concat(ae_output_list, axis=0)
#             # return output[0], ae_output
#             return output, ae_output
#         else:
#             for i in range(self.batch_size):
#                 temp_input = inputs[i, :, :, :, :]
#                 temp_feature = self.encoder(temp_input)
#                 temp_feature = tf.reshape(
#                     temp_feature, (1, self.num_instances, self.num_features))

#                 input_list.append(temp_input)
#                 feature_list.append(temp_feature)
#             feature_concatenated = tf.concat(feature_list, axis=0)
#             feature_distributions = self.kde(
#                 feature_concatenated, self.num_bins, self.sigma)
#             drn_output = self.drn(feature_distributions)
#             output = self.ucc_classifier(
#                 drn_output) if self.ucc_classifier else drn_output
#             # output = GumbelSoftmax()(output, tau=self.gumbel_tau)
#             # return output[0]
#             return output

#     def compute_loss(self, inputs=None, labels=None, outputs=None, reconstruction=None):
#         if self.alpha == 1:
#             assert not isinstance(outputs, type(
#                 None)), "Output classes must be provided"
#             cce = tf.keras.losses.CategoricalCrossentropy()
#             return cce(labels, outputs)
#         elif self.alpha == 0:
#             mse = tf.keras.losses.MeanSquaredError()
#             return mse(reconstruction, inputs)
#         else:
#             if reconstruction != None:
#                 cce = tf.keras.losses.CategoricalCrossentropy()
#                 mse = tf.keras.losses.MeanSquaredError()
#                 return cce(labels, outputs), mse(reconstruction, inputs)
#             else:
#                 cce = tf.keras.losses.CategoricalCrossentropy()
#                 return cce(labels, outputs)

#     def reconstruct_image(self, features):
#         assert self.decoder != None, "Model does not have a decoder"
#         return self.decoder(features)

#     def get_feature_distribution(self, inputs):
#         feature_list = list()
#         for i in range(inputs.shape[0]):
#             temp_input = tf.expand_dims(inputs[i, :, :, :], axis=0)
#             temp_feature = self.encoder(temp_input)
#             temp_feature = tf.reshape(
#                 temp_feature, (1, self.num_instances, self.num_features))
#             feature_list.append(temp_feature)
#         feature_concatenated = tf.concat(feature_list, axis=0)
#         return self.kde(feature_concatenated, self.num_nodes, self.sigma)

#     def ucc_classify(self, inputs):
#         input_list = list()
#         feature_list = list()
#         for i in range(self.batch_size):
#             temp_input = inputs[i, :, :, :, :]
#             temp_feature = self.encoder(temp_input)
#             temp_feature = tf.reshape(
#                 temp_feature, (1, self.num_instances, self.num_features))

#             input_list.append(temp_input)
#             feature_list.append(temp_feature)
#         feature_concatenated = tf.concat(feature_list, axis=0)
#         feature_distributions = self.kde(
#             feature_concatenated, self.num_bins, self.sigma)
#         drn_output = self.drn(feature_distributions)
#         output = self.ucc_classifier(
#             drn_output) if self.ucc_classifier else drn_output
#         # output = GumbelSoftmax()(output, tau=self.gumbel_tau)
#         # return output[0]
#         return output

#     def autoencode(self, inputs):
#         # not sure if we need this or not. Could be useful but not needed atm.
#         pass

#     def extract_features(self, inputs):
#         steps = self.batch_size*self.num_instances
#         length = inputs.shape[0]
#         remainder = length % steps
#         feature_list = []
#         for i in range(np.ceil(length/steps).astype(np.int32)):
#             start = i*steps
#             if i+1 > np.ceil(length/steps):
#                 stop = start + remainder
#             else:
#                 stop = (i+1)*steps
#             input_array = inputs[start:stop, :, :, :]
#             feature_list.append(self.encoder(input_array))
#         feature_concatenated = tf.concat(feature_list, axis=0)
#         return feature_concatenated
