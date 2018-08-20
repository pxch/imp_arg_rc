import torch
from torch import nn
import math
import torch.nn.functional as F


def _masked_softmax(logit, mask, dim=0, eps=1e-10):
    stable_logit = logit - logit.max(dim=dim, keepdim=True)[0]
    exp_logit = torch.exp(stable_logit)
    masked_exp_logit = exp_logit * mask.float() + eps
    masked_exp_sum = masked_exp_logit.sum(dim=dim, keepdim=True)
    masked_softmax = masked_exp_logit / masked_exp_sum
    return masked_softmax


class Attention(nn.Module):
    def __init__(self, method, hidden_size, rescale=False):
        super().__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.rescale = rescale

        if self.rescale:
            self.energy_gain = 1.0 / math.sqrt(self.hidden_size)
        else:
            self.energy_gain = 1.0

        if self.method == 'general':
            self.weight = nn.Linear(
                in_features=self.hidden_size,
                out_features=self.hidden_size,
                bias=False
            )
        elif self.method == 'concat':
            self.weight = nn.Linear(
                in_features=self.hidden_size * 2,
                out_features=self.hidden_size,
                bias=False
            )
            self.vec = nn.Linear(
                in_features=self.hidden_size,
                out_features=1,
                bias=False
            )
        elif self.method == 'dot':
            pass
        else:
            raise NotImplementedError(
                '{} attention not implemented'.format(method))

    # hidden: B * h
    # encoder_outputs: L * B * h
    # softmax_mask: L * B
    # return: L * B
    def forward(self, hidden, encoder_outputs, softmax_mask=None,
                return_energy=False):
        if self.method == 'dot':
            # matmul: B * 1 * h, L * B * h * 1 -> L * B * 1 * 1 -> L * B
            energy = \
                hidden.unsqueeze(1).matmul(
                    encoder_outputs.unsqueeze(3)).squeeze()
            energy = energy * self.energy_gain
        elif self.method == 'general':
            # matmul: B * 1 * h, L * B * h * 1 -> L * B * 1 * 1 -> L * B
            energy = \
                hidden.unsqueeze(1).matmul(
                    self.weight(encoder_outputs).unsqueeze(3)). \
                    squeeze(2).squeeze(2)
            energy = energy * self.energy_gain
        else:
            # cat: L * B * h, L * B * h, dim=2 -> L * B * 2h
            concat = torch.cat(
                (hidden.expand_as(encoder_outputs), encoder_outputs), dim=2)
            # L * B * 2h -> L * B * h -> L * B * 1 -> L * B
            energy = self.vec(torch.tanh(self.weight(concat))).squeeze(2)

        if return_energy:
            if softmax_mask is not None:
                return energy * softmax_mask.float()
            else:
                return energy

        if softmax_mask is not None:
            attn = _masked_softmax(energy, softmax_mask, dim=0)
        else:
            attn = F.softmax(energy, dim=0)
        return attn


class SelfAttention(nn.Module):
    def __init__(self, method, hidden_size, rescale=False):
        super().__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.rescale = rescale

        if self.rescale:
            self.energy_gain = 1.0 / math.sqrt(self.hidden_size)
        else:
            self.energy_gain = 1.0

        if self.method == 'general':
            self.weight = nn.Linear(
                in_features=self.hidden_size,
                out_features=self.hidden_size,
                bias=False
            )
        elif self.method == 'concat':
            self.weight = nn.Linear(
                in_features=self.hidden_size * 2,
                out_features=self.hidden_size,
                bias=False
            )
            self.vec = nn.Linear(
                in_features=self.hidden_size,
                out_features=1,
                bias=False
            )
        elif self.method == 'dot':
            pass
        else:
            raise NotImplementedError(
                '{} attention not implemented'.format(method))

    # encoder_outputs: L * B * h
    # softmax_mask: B * L * L
    # return: L * B * h
    def forward(self, encoder_outputs, softmax_mask):
        max_len = encoder_outputs.size(0)

        if self.method == 'dot':
            # bmm: B * L * h, B * h * L -> B * L * L
            energy = torch.bmm(
                encoder_outputs.transpose(0, 1),
                encoder_outputs.transpose(0, 1).transpose(1, 2))
            energy = energy * self.energy_gain
        elif self.method == 'general':
            # bmm: B * L * h, B * h * L -> B * L * L
            energy = torch.bmm(
                encoder_outputs.transpose(0, 1),
                self.weight(encoder_outputs).transpose(0, 1).transpose(1, 2))
            energy = energy * self.energy_gain
        else:
            # cat: B * L * L * h, B * L * L * h, dim=2 -> B * L * L * 2h
            concat = torch.cat([
                encoder_outputs.transpose(0, 1).unsqueeze(1).expand(
                    [-1, max_len, -1, -1]),
                encoder_outputs.transpose(0, 1).unsqueeze(2).expand(
                    [-1, -1, max_len, -1])], dim=3)
            # B * L * L * 2h -> B * L * L * h -> B * L * L * 1 -> B * L * L
            energy = self.vec(torch.tanh(self.weight(concat))).squeeze(3)

        attn = _masked_softmax(energy, softmax_mask, dim=2)

        # attn = F.softmax(torch.log(torch.exp(energy) * softmax_mask), dim=2)

        return attn
