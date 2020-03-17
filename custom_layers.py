import torch
from torch import nn
from torch.autograd import Function

"""
There are 4 approaches in a 2x2 table:
Rows are about how you compute the "modulating signal" (modsig):

    row1. Compute the forward pass energy and use that as modulating signal (modsig). This has an unsupervised vibe, because it is 
    using other neuron signals to predict in a sense what the gradients would be in the backward pass using their consensus.
    
    row2. Instead of computing the energy in forward pass use the gradients received in the backward pass

Columns are about how you use the modsig in the backward pass:

    col1. Set the weights' gradients to zero by a function of modsig (e.g. probabilistic like dropout or deterministic)
    Here the weights and their input neurons will not receive a training signal. 
    
    col2. Train the weights but don't pass the gradient to the input neurons

I'll like to try row1-col2 combinations first.

The next decision is about how to use the modsig to dropout weights. I can think of two ways:
    1. Drop out the top-k weights and/or input neurons
    2. Compute Bernoulli probabilities using modsig and sample from that distribution to determine what to drop out  

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
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[3]:
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
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
        output = torch.nn.functional.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        ctx.save_for_backward(input, weight, bias, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        #import pdb
        #pdb.Pdb(nosigint=True).set_trace()
        #import pydevd
        #pydevd.settrace(suspend=True, trace_only_current_thread=True)
        input, weight, bias, output = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        grad_input = grad_weight = grad_bias = grad_stride = grad_padding = grad_dilation = grad_groups = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3))  # todo: double check
        return grad_input, grad_weight, grad_bias, grad_stride, grad_padding, grad_dilation, grad_groups


class Conv2DCustom(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2DCustom, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                           dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return Conv2DFunctionCustom.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels, self.out_channels, self.kernel_size)
