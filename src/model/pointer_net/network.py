import torch
from torch import nn

from .attention import Attention
from .encoder import Encoder, SelfAttentiveEncoder

num_salience_features = 4


class SalienceEmbedding(nn.Module):
    def __init__(self, salience_vocab_size, salience_embedding_size):
        super().__init__()

        self.salience_vocab_size = salience_vocab_size
        self.salience_embedding_size = salience_embedding_size
        self.embedding_total = nn.Embedding(
            num_embeddings=self.salience_vocab_size,
            embedding_dim=self.salience_embedding_size)
        self.embedding_named = nn.Embedding(
            num_embeddings=self.salience_vocab_size,
            embedding_dim=self.salience_embedding_size)
        self.embedding_nominal = nn.Embedding(
            num_embeddings=self.salience_vocab_size,
            embedding_dim=self.salience_embedding_size)
        self.embedding_pronominal = nn.Embedding(
            num_embeddings=self.salience_vocab_size,
            embedding_dim=self.salience_embedding_size)

    # num_mentions_total: Ld * B
    # num_mentions_named: Ld * B
    # num_mentions_nominal: Ld * B
    # num_mentions_pronominal: Ld * B
    # return: Ld * B * (4 * salience_embedding_size)
    def forward(self, num_mentions_total, num_mentions_named,
                num_mentions_nominal, num_mentions_pronominal):
        total_embed = self.embedding_total(num_mentions_total)
        named_embed = self.embedding_named(num_mentions_named)
        nominal_embed = self.embedding_nominal(num_mentions_nominal)
        pronominal_embed = self.embedding_pronominal(num_mentions_pronominal)
        return torch.cat(
            [total_embed, named_embed, nominal_embed, pronominal_embed], dim=2)


class PointerNet(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_layers=1,
                 query_num_layers=None, bidirectional=True, dropout_p=0.1,
                 use_self_attention=False, attn_method='general',
                 rescale_attn_energy=False, use_salience=False,
                 salience_vocab_size=None, salience_embedding_size=None,
                 multi_hop=False, extra_query_linear=False,
                 extra_doc_encoder=False, query_aware=False):
        super().__init__()

        # set hyperparameters
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if not query_num_layers:
            self.query_num_layers = num_layers
        else:
            self.query_num_layers = query_num_layers

        self.bidirectional = bidirectional
        self.dropout_p = dropout_p
        self.use_self_attention = use_self_attention
        self.attn_method = attn_method
        self.rescale_attn_energy = rescale_attn_energy

        # create embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.input_size
        )

        self.use_salience = use_salience
        self.salience_vocab_size = salience_vocab_size
        self.salience_embedding_size = salience_embedding_size

        # use 2 layers of attention, the first layer used to compute a weighted
        # sum of document encoder states (o_1) to update the query vector (q_1),
        # and the second layer used to compute pointer probability
        self.multi_hop = multi_hop
        # use an extra linear mapping in updating query vector, i.e.,
        # q_2 = H * q_1 + o_1, instead of q_2 = q_1 + o_1
        self.extra_query_linear = extra_query_linear
        # use an extra layer of document encoder for the second layer of
        # attention, with the hidden states of previous layer as input
        self.extra_doc_encoder = extra_doc_encoder
        # concatenate the query vector to the input of every time step to the
        # extra layer of document encoder, to make it query aware
        self.query_aware = query_aware

        self.salience_input_size = 0

        if self.use_salience:
            if self.salience_vocab_size and self.salience_embedding_size:
                self.salience_embedding = SalienceEmbedding(
                    salience_vocab_size=self.salience_vocab_size,
                    salience_embedding_size=self.salience_embedding_size
                )
                self.salience_input_size = \
                    num_salience_features * salience_embedding_size
            else:
                self.salience_input_size = num_salience_features

        # create document encoder and query encoder
        if self.use_self_attention:
            self.doc_encoder = SelfAttentiveEncoder(
                input_size=self.input_size + self.salience_input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                dropout_p=self.dropout_p,
                attn_method=self.attn_method,
                rescale_attn_energy=self.rescale_attn_energy
            )
        else:
            self.doc_encoder = Encoder(
                input_size=self.input_size + self.salience_input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                dropout_p=self.dropout_p
            )

        self.query_encoder = Encoder(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.query_num_layers,
            bidirectional=self.bidirectional,
            dropout_p=self.dropout_p
        )

        # create attention layer
        attn_hidden_size = \
            self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.attention = Attention(
            method=self.attn_method,
            hidden_size=attn_hidden_size,
            rescale=self.rescale_attn_energy)

        if self.multi_hop:
            self.attention_1 = Attention(
                method=self.attn_method,
                hidden_size=attn_hidden_size,
                rescale=self.rescale_attn_energy)
            if self.extra_query_linear:
                self.query_linear_mapping = nn.Linear(
                    in_features=attn_hidden_size,
                    out_features=attn_hidden_size,
                    bias=False
                )
            if self.extra_doc_encoder:
                if self.query_aware:
                    self.doc_encoder_2 = Encoder(
                        input_size=attn_hidden_size*2,
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        bidirectional=self.bidirectional,
                        dropout_p=self.dropout_p
                    )
                else:
                    self.doc_encoder_2 = Encoder(
                        input_size=attn_hidden_size,
                        hidden_size=self.hidden_size,
                        num_layers=self.num_layers,
                        bidirectional=self.bidirectional,
                        dropout_p=self.dropout_p
                    )

    # initialize embedding layer with pre-trained word vectors
    # vectors: |V| * input_size
    def init_embedding(self, vectors: torch.FloatTensor):
        assert (self.vocab_size, self.input_size) == vectors.shape
        self.embedding.weight = nn.Parameter(vectors)

    # query_input_seqs: Lq * B (unsorted)
    # query_input_lengths: B (LongTensor)
    # return: B * h (or B * 2h for bidirectional)
    def get_query_hidden(self, query_input_seqs, query_input_lengths):
        sorted_query_input_lengths, sorted_query_indices = \
            torch.sort(query_input_lengths, dim=0, descending=True)

        sorted_query_input_seqs = query_input_seqs[:, sorted_query_indices]
        sorted_query_input_embeddings = self.embedding(sorted_query_input_seqs)

        _, sorted_query_hidden = self.query_encoder(
            sorted_query_input_embeddings, sorted_query_input_lengths)

        _, reverse_indices = torch.sort(sorted_query_indices, dim=0)

        query_hidden = sorted_query_hidden[reverse_indices]

        return query_hidden

    # document_input_seqs: Ld * B
    # document_input_lengths: B
    # query_input_seqs: Lq * B
    # query_input_lengths: B
    # softmax_mask: Ld * B
    # return attn: Ld * B
    # return self_attn: B * Ld * Ld or None
    # return first_hop_attn: Ld * B or None
    # return neg_attn: Ld * B or None
    def forward(self, doc_input_seqs, doc_input_lengths,
                query_input_seqs, query_input_lengths, softmax_mask,
                return_energy=False, multi_hop=False, event_attention=False,
                **kwargs):
        doc_input_embeddings = self.embedding(doc_input_seqs)

        if self.use_salience:
            if self.salience_vocab_size and self.salience_embedding_size:
                doc_salience_embeddings = self.salience_embedding(
                    kwargs['num_mentions_total'],
                    kwargs['num_mentions_named'],
                    kwargs['num_mentions_nominal'],
                    kwargs['num_mentions_pronominal']
                )
            else:
                doc_salience_embeddings = torch.cat([
                    kwargs['num_mentions_total'].unsqueeze(2),
                    kwargs['num_mentions_named'].unsqueeze(2),
                    kwargs['num_mentions_nominal'].unsqueeze(2),
                    kwargs['num_mentions_pronominal'].unsqueeze(2)], dim=2
                ).float()
            doc_input_embeddings = torch.cat(
                [doc_input_embeddings, doc_salience_embeddings], dim=2)

        if self.use_self_attention:
            doc_outputs, _, self_attn = self.doc_encoder(
                doc_input_embeddings, doc_input_lengths)

        else:
            doc_outputs, _ = self.doc_encoder(
                doc_input_embeddings, doc_input_lengths)
            self_attn = None

        query_hidden = self.get_query_hidden(
            query_input_seqs=query_input_seqs,
            query_input_lengths=query_input_lengths)

        if multi_hop:
            # if event_attention:
            #     event_forward_indices = kwargs['event_forward_indices']
            #     event_backward_indices = kwargs['event_backward_indices']
            #     event_outputs = torch.cat([
            #         doc_outputs[:, :, :self.hidden_size].index_select(
            #             event_forward_indices, dim=0),
            #         doc_outputs[:, :, self.hidden_size:].index_select(
            #             event_backward_indices, dim=0)], dim=2)
            #
            #     attn_1 = self.attention_1(query_hidden, event_outputs)
            #
            #     output_1 = torch.bmm(
            #         attn_1.transpose(0, 1).unsqueeze(1),
            #         event_outputs.transpose(0, 1)).squeeze(1)
            # else:
            first_hop_attn = self.attention_1(query_hidden, doc_outputs)

            # bmm: B * 1 * L, B * L * 2h -> B * 1 * 2h -> B * 2h
            output_1 = torch.bmm(
                first_hop_attn.transpose(0, 1).unsqueeze(1),
                doc_outputs.transpose(0, 1)).squeeze(1)

            if self.extra_query_linear:
                query_hidden_2 = \
                    self.query_linear_mapping(query_hidden) + output_1
            else:
                query_hidden_2 = query_hidden + output_1

            if self.extra_doc_encoder:
                if self.query_aware:
                    max_len = doc_outputs.size(0)
                    query_aware_inputs = \
                        torch.cat([doc_outputs,
                                   query_hidden.expand(max_len, -1, -1)], dim=2)
                    doc_outputs_2, _ = self.doc_encoder_2(
                        query_aware_inputs, doc_input_lengths)
                else:
                    doc_outputs_2, _ = self.doc_encoder_2(
                        doc_outputs, doc_input_lengths)
                attn = self.attention(
                    query_hidden_2, doc_outputs_2, softmax_mask,
                    return_energy=return_energy)
            else:
                attn = self.attention(
                    query_hidden_2, doc_outputs, softmax_mask,
                    return_energy=return_energy)
        else:
            first_hop_attn = None
            attn = self.attention(
                query_hidden, doc_outputs, softmax_mask,
                return_energy=return_energy)

        if 'neg_query_input_seqs' in kwargs and \
                'neg_query_input_lengths' in kwargs:

            neg_query_hidden = self.get_query_hidden(
                query_input_seqs=kwargs['neg_query_input_seqs'],
                query_input_lengths=kwargs['neg_query_input_lengths'])

            neg_attn = self.attention(
                neg_query_hidden, doc_outputs, softmax_mask,
                return_energy=return_energy)

        else:
            neg_attn = None

        return attn, self_attn, first_hop_attn, neg_attn
