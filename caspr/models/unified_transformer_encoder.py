"""CASPR Transfomer model."""

import warnings

import torch
import torch.nn as nn

from caspr.models.embedding_layer import CategoricalEmbedding

warnings.simplefilter('ignore')

SEQ_CAT_INDEX = 0
SEQ_CONT_INDEX = 1
NON_SEQ_CAT_INDEX = 2
NON_SEQ_CONT_INDEX = 3


class UnifiedTransformerEncoder(nn.Module):  # noqa: R0902, W0223
    """Encapsulates the basic structure to run most of our models.

    It checks the various conditions for the presence or the absence of data
    and is compatible with functionalities like
            1. Usage of pretrained embedding vectors
            2. Multi-Layered transformer use
            3. Convolutional Aggregation for data
            4. Self-Multi-Head and Bahdanau Attention (when number of heads = 1, Bahdanau is used by default)

        In this new edition, it is compatible with the DLExplainer module and should be used if explainability
        is a requirement
    """

    def __init__(self,  # noqa: R0913, R0914
                 transformer_encoder,
                 emb_dims_non_seq,
                 emb_dropout_non_seq,
                 emb_dims_seq,
                 emb_dropout_seq,
                 hidden_size,
                 seq_cont_count,
                 non_seq_cont_count,
                 non_seq_pretrained_embs=None,
                 freeze_non_seq_pretrained_embs=True,
                 seq_pretrained_embs=None,
                 freeze_seq_pretrained_embs=True):
        """Initialize model with params."""

        super().__init__()

        self._explain = False

        self.emb_non_seq = CategoricalEmbedding(emb_dims=emb_dims_non_seq, emb_dropout=emb_dropout_non_seq,
                                                is_seq=False, pretrained_vecs=non_seq_pretrained_embs,
                                                freeze_pretrained=freeze_non_seq_pretrained_embs)
        self.emb_seq = CategoricalEmbedding(emb_dims=emb_dims_seq, emb_dropout=emb_dropout_seq,
                                            is_seq=True, pretrained_vecs=seq_pretrained_embs,
                                            freeze_pretrained=freeze_seq_pretrained_embs)

        self.hid_dim = hidden_size
        self.seq_cont_dim = seq_cont_count
        self.non_seq_cont_dim = non_seq_cont_count

        # Linear layers for seq_data
        seq_inp_size = self.emb_seq.emb_size + self.seq_cont_dim
        # print('self.linear_seq self.emb_seq.emb_size', self.emb_seq.emb_size)
        # print('self.linear_seq self.seq_cont_dim', self.seq_cont_dim)
        # print('self.linear_seq seq_inp_size', seq_inp_size)
        # print('self.linear_seq self.hid_dim', self.hid_dim)
        self.linear_seq = nn.Linear(seq_inp_size, self.hid_dim)

        # Linear layers for non_seq_data
        non_seq_inp_size = self.emb_non_seq.emb_size + self.non_seq_cont_dim
        # print('non_seq_inp_size, self.hid_dim: ', non_seq_inp_size, self.hid_dim)
        self.linear_non_seq = nn.Linear(non_seq_inp_size, self.hid_dim) if non_seq_inp_size else None

        self.transformer_encoder = transformer_encoder

    def forward(self, key_pad_mask=None, *args):
        """Run a forward pass of model over the data."""

        nonempty_idx = args[-1]
        # key_pad_mask = args[1]
        # print('key_pad_mask.shape 4: ', key_pad_mask.shape)
        data_exists = list(map(lambda x: x != -1, nonempty_idx))
        device = args[0].device
        batch_size, seq_len = args[0].shape[:2]
        # print('batch_size, seq_len: ', batch_size, seq_len)

        seq_cat_data = args[nonempty_idx[SEQ_CAT_INDEX]] if data_exists[SEQ_CAT_INDEX] else torch.empty(batch_size, seq_len, 0, device=device)
        seq_cont_data = args[nonempty_idx[SEQ_CONT_INDEX]] if data_exists[SEQ_CONT_INDEX] else torch.empty(batch_size, seq_len, 0, device=device)
        non_seq_cat_data = args[nonempty_idx[NON_SEQ_CAT_INDEX]] if data_exists[NON_SEQ_CAT_INDEX] else torch.empty(batch_size, 0, device=device)
        non_seq_cont_data = args[nonempty_idx[NON_SEQ_CONT_INDEX]] if data_exists[NON_SEQ_CONT_INDEX] else torch.empty(batch_size, 0, device=device)
        
        # print('seq_cont_data.shape: ', seq_cont_data.shape)
        # print('seq_cont_data[:3, :, :]: ', seq_cont_data[:3, :, :])

        if self.emb_seq and data_exists[SEQ_CAT_INDEX]:
            seq_cat_data = self.emb_seq(seq_cat_data)
        
        seq_inp = torch.cat((seq_cat_data, seq_cont_data), -1)
        # print('seq_inp.shape 1: ', seq_inp.shape)
        # print('seq_cat_data.shape', seq_cat_data.shape)
        # print('seq_cont_data.shape', seq_cont_data.shape)
        # print('seq_inp.shape', seq_inp.shape)
        seq_inp = self.linear_seq(seq_inp)
        # print('seq_inp.shape 2: ', seq_inp.shape)

        if self.emb_non_seq and data_exists[NON_SEQ_CAT_INDEX]:
            non_seq_cat_data = self.emb_non_seq(non_seq_cat_data)
        non_seq_inp = torch.cat((non_seq_cat_data, non_seq_cont_data), -1)
        # print('non_seq_inp.shape 1: ', non_seq_inp.shape)
        if self.linear_non_seq:
            non_seq_inp = self.linear_non_seq(non_seq_inp).unsqueeze(1)
        # print('non_seq_inp.shape 2: ', non_seq_inp.shape)
        
        src_inp = torch.cat((seq_inp, non_seq_inp), 1) if non_seq_inp.nelement() > 0 else seq_inp
        # src_inp = [batch_size, src len, hid dim]
        # print('src_inp.shape 1: ', src_inp.shape)
        non_seq_pad_mask = torch.ones(key_pad_mask.shape[0], 1, 1, 1, device=device).byte()
        key_padding_mask = torch.cat((key_pad_mask, non_seq_pad_mask), 3)
        # print('key_pad_mask', key_pad_mask[:2, :, :, :5])
        # print('non_seq_pad_mask.shape 1: ', non_seq_pad_mask.shape)
        # print('key_pad_mask.shape 5: ', key_padding_mask.shape)

        enc_src, src_mask = self.transformer_encoder(src_inp, key_padding_mask)

        if self._explain:
            return enc_src.reshape(enc_src.shape[0], -1)
        return enc_src, src_mask, src_inp

    @property
    def explain(self):
        """Getter for explain."""

        return self._explain

    def set_explain(self, value):
        """Setter for explain."""

        self._explain = value
