import torch
from torch import nn
from torch.autograd import Function
from conf import global_conf as conf

"""
There are 4 approaches in a 2x2 table:
Rows are about how you compute the "modulating signal" (modsig):

    row1. Compute the forward pass energy and use that as modulating signal (modsig). This has an unsupervised vibe, because it is 
    using other neuron signals to predict in a sense what the gradients would be in the backward pass using their consensus.
    This could be done on logits or activations.
    
    row2. Instead of computing the energy in forward pass use the gradients received in the backward pass

Columns are about how you use the modsig in the backward pass:

    col1. Set the weights' gradients to zero by a function of modsig (e.g. probabilistic like dropout or deterministic)
    Here the weights and their input neurons will not receive a training signal. 
    
    col2. Train the weights but don't pass the gradient to the input neurons. I feel like doing this with a large L1 
    regularization term, could result in sparse but "rich"(orthonogal?) functions.

I'll like to try row2-col2 combinations first.

The next decision is about how to use the modsig to dropout weights. I can think of two ways:
    1. Drop out the top-k weights and/or input neurons
    2. Compute Bernoulli probabilities using modsig and sample from that distribution to determine what to drop out  


Experiment idea: Try transfer learning with across datasets, or classes. For example, train on CIFAR10 and use the 
learned features on CIFAR100, just training the last layer. 

Idea: Remove activation functions (e.g. Relu layers) altogether, and do forward pass dropout using the "consensus" mod sig.
This is interesting, because there is no explicit non-linearity like a normal neural network, but there is a different
type of non-linearity that stems from dropping nodes(or edges?) out. At the same time the whole system is trainable with
backprop. This is a Frankenstein hybrid between my IGS model (which is an EBM) and deep learning. If this performs 
comparably to deep learning it would be a big leap for my s/IGS. 
Also there is no reason the energy should be calculated only across one layer. However, doing it across multiple layers
makes it slower, and makes it look like Gibbs sampling of EBMs. 

"""




# Inherit from Function
class LinearFunctionCustom(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # import pydevd
        # pydevd.settrace(suspend=True, trace_only_current_thread=True)
        input, weight, bias, output = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if ctx.needs_input_grad[0]:
            if conf['topk']:
                # todo: do the inverse top-k as an experiment
                topk_idx = grad_weight.abs().topk(int(grad_weight.shape[1] * conf['topk_ratio']), dim=1)[1]
                topk_mask = torch.ones((weight.shape[0], weight.shape[1]), device=weight.device).scatter(1, topk_idx, 0.)
                bw_weight = weight * topk_mask
            else:
                bw_weight = weight
            grad_input = grad_output.mm(bw_weight)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

class LinearCustom(nn.Linear):
    def __init__(self, input_features, output_features, bias=True, conf=12):
        super(LinearCustom, self).__init__(input_features, output_features, bias=True)

    def forward(self, input):
        return LinearFunctionCustom.apply(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Conv2DFunctionCustom(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
                cum_grad_w=None, cum_grad_b=None):
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        output = torch.nn.functional.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        ctx.save_for_backward(input, weight, bias, output, cum_grad_w, cum_grad_b)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        #import pdb
        #pdb.Pdb(nosigint=True).set_trace()
        # import pydevd
        # pydevd.settrace(suspend=True, trace_only_current_thread=True)
        input, weight, bias, output, cum_grad_w, cum_grad_b = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        grad_input = grad_weight = grad_bias = grad_stride = grad_padding = grad_dilation = grad_groups = None
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups).contiguous()
            cum_grad_w += torch.abs(grad_weight)
        if ctx.needs_input_grad[0]:
            if conf['topk']:
                # todo: do the inverse top-k as an experiment
                topk_idx = grad_weight.abs().sum((2, 3)).topk(int(grad_weight.shape[1] * conf['topk_ratio']), dim=1)[1]
                topk_mask = torch.ones((weight.shape[0], weight.shape[1]), device=weight.device).scatter(1, topk_idx, 0.)
                topk_mask = topk_mask.unsqueeze(-1).unsqueeze(-1)
                bw_weight = weight * topk_mask
            else:
                bw_weight = weight
            grad_input = torch.nn.grad.conv2d_input(input.shape, bw_weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3))  # todo: double check
            cum_grad_b += torch.abs(grad_bias)
        return grad_input, grad_weight, grad_bias, grad_stride, grad_padding, grad_dilation, grad_groups, None, None


class Conv2DCustom(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, name=None):
        super(Conv2DCustom, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                           dilation=dilation, groups=groups, bias=bias)
        self.cum_grad_w = torch.nn.Parameter(torch.zeros_like(self.weight))
        self.cum_grad_b = torch.nn.Parameter(torch.zeros_like(self.bias))
        if name:
            self.w_logger = WandBLogger(name=name+'_w')
            self.b_logger = WandBLogger(name=name+'_b')

    def forward(self, x):
        out = Conv2DFunctionCustom.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
                                          self.cum_grad_w, self.cum_grad_b)
        # logging cumulative absolute gradients
        if hasattr(self, 'w_logger') and hasattr(self, 'b_logger'):
            self.w_logger(self.cum_grad_w)
            self.b_logger(self.cum_grad_b)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels, self.out_channels, self.kernel_size)


class WandBLogger(nn.Module):
    """
    A module for logging values into W&B
    `global_conf['wandb']` needs to be set for this to work
    """
    def __init__(self, name, frac_zero=False):
        super(WandBLogger, self).__init__()
        self.name = str(name)
        self.frac_zero = frac_zero

    def forward(self, x):
        if conf['wandb'] is not None and conf['log_intermediate']:
            train_val = '_t' if self.training else '_v'
            conf['wandb'].log({self.name+train_val: x.cpu()}, step=conf['epoch'])
            if self.frac_zero:
                zeros_fraction = (x == 0.0).float().mean()
                conf['wandb'].log({self.name+'_frac_zeros_'+train_val: zeros_fraction.cpu()}, step=conf['epoch'])
        return x

    def extra_repr(self):
        return 'Logging name={}'.format(self.name)
