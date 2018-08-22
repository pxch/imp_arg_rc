import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .attention import SelfAttention


class Encoder(nn.Module):
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
            dropout=self.dropout_p if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )

    # input_embeddings: L * B * d (embedding dimension)
    # outputs: L * B * h (or L * B * 2h for bidirectional)
    # final_hidden: B * h (or B * 2h for bidirectional)
    def forward(self, input_embeddings, input_lengths, hidden=None):
        packed = pack_padded_sequence(
            self.dropout(input_embeddings), list(input_lengths))
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(outputs)
        if self.bidirectional:
            final_hidden = torch.cat(
                [hidden[-2, :, :], hidden[-1, :, :]], dim=1)
        else:
            final_hidden = hidden[-1, :, :]
        return outputs, final_hidden


class SelfAttentiveEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,
                 bidirectional=True, dropout_p=0.1, attn_method='general',
                 rescale_attn_energy=False):
        super().__init__()

        # set hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_p = dropout_p

        assert attn_method in ['general', 'concat', 'dot']
        self.attn_method = attn_method
        self.rescale_attn_energy = rescale_attn_energy

        # create the dropout layer for embedding input
        self.dropout = nn.Dropout(p=self.dropout_p)

        # create the first rnn layers
        self.gru_0 = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bidirectional=self.bidirectional
        )

        if self.bidirectional:
            self.output_size = self.hidden_size * 2
        else:
            self.output_size = self.hidden_size

        # create subsequent self-attention and rnn layers
        for layer_idx in range(1, self.num_layers):
            attn_layer_name = 'attn_{}'.format(layer_idx)
            attn_layer = SelfAttention(
                method=self.attn_method,
                hidden_size=self.output_size,
                rescale=self.rescale_attn_energy
            )
            setattr(self, attn_layer_name, attn_layer)

            gru_layer_name = 'gru_{}'.format(layer_idx)

            gru_layer = nn.GRU(
                input_size=self.output_size,
                hidden_size=self.hidden_size,
                num_layers=1,
                bidirectional=self.bidirectional
            )
            setattr(self, gru_layer_name, gru_layer)

    @staticmethod
    def get_mask_for_self_attention(input_lengths, max_len):
        indices = torch.arange(0, max_len).expand(max_len, -1).unsqueeze(0)
        indices = indices.to(input_lengths)
        mask = indices.lt(input_lengths.unsqueeze(1).unsqueeze(2)).float()
        indices_2 = torch.arange(0, max_len).view(-1, 1).expand(
            max_len, max_len).unsqueeze(0)
        indices_2 = indices_2.to(input_lengths)
        mask_2 = indices_2.lt(input_lengths.unsqueeze(1).unsqueeze(2)).float()
        mask = mask * mask_2
        # we should allow self-attention to put some weight on itself, right?
        # mask = (mask - torch.eye(max_len).type_as(mask)).clamp(min=0)
        return mask

    # inputs: L * B * d
    # input_lengths: B (LongTensor)
    # outputs: L * B * h (or L * B * 2h for bidirectional)
    # final_hidden: B * h (or B * 2h for bidirectional)
    @staticmethod
    def forward_gru_layer(gru, inputs, input_lengths, hidden=None):
        packed = pack_padded_sequence(inputs, list(input_lengths))
        outputs, hidden = gru(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(outputs)
        if gru.bidirectional:
            final_hidden = torch.cat(
                [hidden[-2, :, :], hidden[-1, :, :]], dim=1)
        else:
            final_hidden = hidden[-1, :, :]
        return outputs, final_hidden

    # input_embeddings: L * B * d
    # input_lengths: B (LongTensor)
    # outputs: L * B * h (or L * B * 2h for bidirectional)
    # final_hidden: B * h (or B * 2h for bidirectional)
    # self_attn: B * L * L
    def forward(self, input_embeddings, input_lengths):
        gru_inputs = self.dropout(input_embeddings)

        gru_outputs, final_hidden = SelfAttentiveEncoder.forward_gru_layer(
            self.gru_0, gru_inputs, input_lengths)

        self_attn = None

        max_len = input_embeddings.size(0)
        self_attention_softmax_mask = self.get_mask_for_self_attention(
            input_lengths, max_len)

        for layer_idx in range(1, self.num_layers):
            attn_layer = getattr(self, 'attn_{}'.format(layer_idx))

            self_attn = attn_layer(gru_outputs, self_attention_softmax_mask)

            attn_outputs = torch.bmm(
                self_attn, gru_outputs.transpose(0, 1)).transpose(0, 1)

            gru_inputs = self.dropout(gru_outputs + attn_outputs)

            gru_layer = getattr(self, 'gru_{}'.format(layer_idx))

            gru_outputs, final_hidden = SelfAttentiveEncoder.forward_gru_layer(
                gru_layer, gru_inputs, input_lengths)

        return gru_outputs, final_hidden, self_attn
