import torch
import torch.nn as nn
import torch.nn.functional as F


class Program(nn.Module):
    def __init__(self, out_size):
        super(Program, self).__init__()
        self.weight = torch.nn.Parameter(data=torch.Tensor(3, *out_size))
        self.weight.data.uniform_(-1, 1)

    def forward(self, x):
        x = self.weight.mul(x)
        return x


class AdvProgram(nn.Module):
    def __init__(self, in_size, out_size, mask_size, device=torch.device('cuda')):
        super(AdvProgram, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.program = Program(out_size).to(device)

        l_pad = int((mask_size[0]-in_size[0]+1)/2)
        r_pad = int((mask_size[0]-in_size[0])/2)

        mask = torch.zeros(3, *in_size, device=device)
        self.mask = F.pad(mask, (l_pad, r_pad, l_pad, r_pad), value=1)

    def forward(self, x):
        x = x + torch.tanh(self.program(self.mask))
        return x
