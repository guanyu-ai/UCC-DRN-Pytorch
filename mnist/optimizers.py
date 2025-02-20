import torch
import numpy as np
from torch._C import Value
from torch.optim import optimizer
from typing import Generator, Iterable

class SGLD(optimizer.Optimizer):
    """Stochastic Gradient Langevin Dynamics (SGLD).
       An algorithm for Bayesian learning from large scale datasets.

       Weight decay is specified in terms of the Gaussian prior's sigma.

       Welling and Teh, 2011. Bayesian Learning via Stochastiv Gradient Langevin
       Dynamics. Paper link: https://bit.ly/3ngnyRA

       Args:
           params (Iterable): an iterable of `torch.Tensor`s or
               `dict`s. Specifies what Tensors should be optimized.
           defaults: (dict): a dict containing default values of optimization
               options (used when a parameter group doesn't specify them).
           lr (float): learning rate.
           sigma_gauss_prior (float, optional): Defaults to 0.
           add_noise (bool, optional): Defaults to True.

       Attributes:
           param_group (OptimizerParamGroup): Stores parameters in the param_group
               and stores a pointer to the OptimizerOptions.
               docs: https://preview.tinyurl.com/2y272xmv

       Parameters need to be specified as collections that have a deterministic
           ordering that is consistent between runs. Examples of objects that don't
           satisfy those properties are sets and iterators over values of
           dictionaries.
       """
    def __init__(
        self,
        params:Iterable,
        lr:float,
        sigma_gauss_prior: float = 0.1,
        add_noise: bool = True
    ) -> None:
        weight_decay = 1/sigma_gauss_prior**2
        defaults = dict(lr=lr, weight_decay=weight_decay, add_noise=add_noise)
        super(SGLD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Updates neural network parameters. Called once the gradients are
        computed using loss.backward(). Performs a single parameter update.

        Args:
            closure (callable, optional): A closure that reevaluates the
                model and returns the loss.
        This function should not modify the gradient field of the parameters,
            i.e. `parameter.grad`.
        """
        loss = None
        def params()-> Generator:
            for param_group in self.param_groups:
                weight_decay = param_group['weight_decay']
                for param in param_group['params']:
                    yield param, weight_decay, param_group

        for param, weight_decay, param_group in params():
            if param.grad is None:
                continue
            gradient = param.grad.data
            if weight_decay !=0:
                gradient.add_(weight_decay, param.data)
            if param_group['addnoise']:
                langevin_noise = param.data.new(param.data.size()).normal_(
                    mean=0, std=1
                )/ np.sqrt(param_group['lr'])
                param.data.add_(-param_group['lr'], 0.5*gradient + langevin_noise)
            else:
                param.data.add_(-param_group['lr'], 0.5*gradient)
        return loss
