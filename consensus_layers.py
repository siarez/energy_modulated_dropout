import torch
from torch import nn
from torch.autograd import Function
from conf import global_conf as conf


class LinearConsensus(nn.Linear):
    def __init__(self, input_features, output_features, bias=True, conf=12):
        super(LinearConsensus, self).__init__(input_features, output_features, bias=True)

    def forward(self, input):
        return LinearFunctionConsensus.apply(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class LinearFunctionConsensus(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        edge_energy = (output.T.mm(input) * weight).detach()
        topk_idx = edge_energy.abs().topk(int(edge_energy.shape[1] * conf['topk_ratio']), dim=1)[1]
        topk_mask = torch.ones((weight.shape[0], weight.shape[1]), device=weight.device).scatter(1, topk_idx, 0.)
        mod_weight = weight * topk_mask
        output_mod = input.mm(mod_weight.t())
        ctx.save_for_backward(input, weight, bias, output_mod)
        return output_mod

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



