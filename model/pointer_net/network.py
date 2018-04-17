import torch
from torch import nn

from .attention import PointerNetAttention
from .encoder import PointerNetEncoder


def _masked_softmax(logit, mask, dim=0, eps=1e-10):
    stable_logit = logit - logit.max(dim=dim, keepdim=True)[0]
    exp_logit = torch.exp(stable_logit)
    masked_exp_logit = exp_logit * mask.float() + eps
    masked_exp_sum = masked_exp_logit.sum(dim=dim, keepdim=True)
    masked_softmax = masked_exp_logit / masked_exp_sum
    return masked_softmax


class PointerNet(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_layers=1,
                 bidirectional=True, dropout_p=0.1, attn_method='general'):
        super().__init__()

        # set hyperparameters
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_p = dropout_p
        self.attn_method = attn_method

        # create embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.input_size
        )

        # create document encoder and query encoder
        self.doc_encoder = PointerNetEncoder(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout_p=self.dropout_p
        )
        self.query_encoder = PointerNetEncoder(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout_p=self.dropout_p
        )

        # create attention layer
        attn_hidden_size = \
            self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.attention = PointerNetAttention(
            method=self.attn_method, hidden_size=attn_hidden_size)

    # initialize embedding layer with pre-trained word vectors
    # vectors: |V| * input_size
    def init_embedding(self, vectors: torch.FloatTensor):
        assert (self.vocab_size, self.input_size) == vectors.shape
        self.embedding.weight = nn.Parameter(vectors)

    # document_input_seqs: Ld * B
    # document_input_lengths: B
    # query_input_seqs: Lq * B
    # query_input_lengths: B
    # softmax_mask: Ld * B
    # return: B * 1 * Ld
    def forward(self, doc_input_seqs, doc_input_lengths,
                query_input_seqs, query_input_lengths, doc_entity_ids):
        doc_input_embeddings = self.embedding(doc_input_seqs)
        doc_outputs, _ = self.doc_encoder(
            doc_input_embeddings, list(doc_input_lengths))

        sorted_query_input_lengths, sorted_query_indices = \
            torch.sort(query_input_lengths, dim=0, descending=True)

        sorted_query_input_seqs = query_input_seqs[:, sorted_query_indices]
        sorted_query_input_embeddings = self.embedding(sorted_query_input_seqs)

        _, sorted_query_hidden = self.query_encoder(
            sorted_query_input_embeddings, list(sorted_query_input_lengths))

        _, reverse_indices = torch.sort(sorted_query_indices, dim=0)

        query_hidden = sorted_query_hidden[reverse_indices]

        attn_energies = self.attention(query_hidden, doc_outputs)

        softmax_mask = (doc_entity_ids != -1)
        attn = _masked_softmax(logit=attn_energies, mask=softmax_mask, dim=0)

        # masked_attn_energies = \
        #     attn_energies.masked_fill((1 - softmax_mask.long()).byte(), -1e10)
        #
        # log_attn = F.log_softmax(masked_attn_energies, dim=0)

        return attn
