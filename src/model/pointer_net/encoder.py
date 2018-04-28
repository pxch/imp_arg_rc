import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PointerNetEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,
                 bidirectional=True, dropout_p=0.1):
        super().__init__()

        # set hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_p = dropout_p

        # create the dropout layer for embedding input
        self.dropout = nn.Dropout(p=dropout_p)

        # create the rnn layer
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_p,
            bidirectional=self.bidirectional
        )

    # input_embeddings: L * B * d (embedding dimension)
    # outputs: L * B * h (or L * B * 2h for bidirectional)
    # final_hidden: B * h (or B * 2h for bidirectional)
    def forward(self, input_embeddings, input_lengths, hidden=None):
        packed = pack_padded_sequence(
            self.dropout(input_embeddings), input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(outputs)
        if self.bidirectional:
            final_hidden = torch.cat(
                [hidden[-2, :, :], hidden[-1, :, :]], dim=1)
        else:
            final_hidden = hidden[-1, :, :]
        return outputs, final_hidden
