from utils import registry
from spikingjelly.clock_driven.neuron import *

MODELS = registry.Registry('models')

class ReLUX(nn.Module):
    def __init__(self, thre=4):
        super(ReLUX, self).__init__()
        self.thre = thre

    def forward(self, input):
        return torch.clamp(input, 0, self.thre)

relu4 = ReLUX(thre=4)

class multispike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens=4):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return torch.floor(relu4(input) + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp1 = 0 < input
        temp2 = input < ctx.lens
        return grad_input * temp1.float() * temp2.float(), None
    
class Multispike(nn.Module):
    def __init__(self, lens=4, spike=multispike):
        super().__init__()
        self.lens = lens
        self.spike = spike

    def forward(self, inputs):
        return self.spike.apply(inputs)
    

def build_model_from_cfg(cfg, **kwargs):
    return MODELS.build(cfg, **kwargs)

def build_spike_node(timestep, spike_mode, tau=2.0, v_threshold=0.5):
    if spike_mode == "lif":
        proj_lif = MultiStepLIFNode(timestep=timestep, v_threshold=v_threshold, tau=tau, detach_reset=True, backend="cupy")
    elif spike_mode == "elif":
        proj_lif = MultiStepEIFNode(timestep=timestep, v_threshold=v_threshold, tau=tau, detach_reset=True, backend="cupy")
    elif spike_mode == "plif":
        proj_lif = MultiStepParametricLIFNode(timestep=timestep, v_threshold=v_threshold, init_tau=tau, detach_reset=True, backend="cupy")
    elif spike_mode == "if":
        proj_lif = MultiStepIFNode(timestep=timestep, v_threshold=v_threshold, detach_reset=True, backend="cupy")
    elif spike_mode == "ilif":
        proj_lif = Multispike()
    elif spike_mode == "manlif":
        proj_lif = MultiStepManLIFNode(timestep=timestep, v_threshold=v_threshold, init_tau=tau, detach_reset=True, backend="torch")
    else:
        raise ValueError(f"Unsupported spike_mode: {spike_mode}")
    return proj_lif