import torch
from torch import nn


class PointerNetAttention(nn.Module):
    def __init__(self, method, hidden_size):
        super().__init__()

        self.method = method
        self.hidden_size = hidden_size

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
    # return: L * B
    def forward(self, hidden, encoder_outputs):
        if self.method == 'dot':
            # matmul: B * 1 * h, L * B * h * 1 -> L * B * 1 * 1 -> L * B
            energy = \
                hidden.unsqueeze(1).matmul(
                    encoder_outputs.unsqueeze(3)).squeeze()
        elif self.method == 'general':
            # matmul: B * 1 * h, L * B * h * 1 -> L * B * 1 * 1 -> L * B
            energy = \
                hidden.unsqueeze(1).matmul(
                    self.weight(encoder_outputs).unsqueeze(3)).squeeze()
        else:
            # cat: L * B * h, L * B * h, dim=2 -> L * B * 2h
            concat = torch.cat(
                (hidden.expand_as(encoder_outputs), encoder_outputs), dim=2)
            # L * B * 2h -> L * B * h -> L * B * 1 -> L * B
            energy = self.vec(torch.tanh(self.weight(concat))).squeeze()

        return energy
