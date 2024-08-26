import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

class TransfomerSyn(nn.Module):
    def __init__(self, obs_size, d_model, d_out, nhead, num_encoder_layers,
                 pool, dim_ff, dropout, layer_norm_eps):
        super(TransfomerSyn, self).__init__()
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers

        self.embedded = nn.Embedding(obs_size, d_model)
        self.transformer = TransformerEncoderModel(d_model=d_model, nhead=nhead,
                                                   num_encoder_layers=num_encoder_layers,
                                                   pool=pool, dim_feedforward=dim_ff,
                                                   dropout=dropout, d_out=d_out,
                                                   layer_norm_eps=layer_norm_eps)

    def forward(self, text):
        embed_text = self.embedded(text)
        feature = self.transformer(embed_text)
        return feature

    def init_by_TFixup(self):

        for p in self.embedded.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, 0, self.d_model ** (- 1. / 2.))

        temp_state_dic = {}
        for name, param in self.embedded.named_parameters():
            if 'weight' in name:
                temp_state_dic[name] = ((9 * self.num_encoder_layers) ** (- 1. / 4.)) * param

        for name in self.embedded.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.embedded.state_dict()[name]
        self.embedded.load_state_dict(temp_state_dic)

        temp_state_dic = {}
        for name, param in self.transformer.named_parameters():
            if any(s in name for s in ["linear1.weight", "linear2.weight", "self_attn.out_proj.weight"]):
                temp_state_dic[name] = (0.67 * (self.num_encoder_layers) ** (- 1. / 4.)) * param
            elif "self_attn.in_proj_weight" in name:
                temp_state_dic[name] = (0.67 * (self.num_encoder_layers) ** (- 1. / 4.)) * (param * (2 ** 0.5))

        for name in self.transformer.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.transformer.state_dict()[name]
        self.transformer.load_state_dict(temp_state_dic)

class TransformerEncoderModel(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 1, pool: str = 'mean',
                 dim_feedforward: int = 2048, dropout: float = 0.1, d_out: int = 8, activation = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False):
        """

        :param d_model: the number of expected features in the encoder/decoder inputs (default=512).
        :param nhead: the number of heads in the multiheadattention models (default=8).
        :param num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        :param dim_feedforward: the dimension of the feedforward network model (default=2048).
        :param dropout: the dropout value (default=0.1).
        :param activation: the activation function of encoder/decoder intermediate layer, can be a string
                           ("relu" or "gelu") or a unary callable. Default: relu
        :param layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        :param batch_first: If ``True``, then the input and output tensors are provided
                            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        :param norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
                           other attention and feedforward operations, otherwise after. Default: ``False``

        Examples::
            >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
            >>> src = torch.rand((10, 32, 512))
            >>> out = transformer_model(src)
        """
        super(TransformerEncoderModel, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first, norm_first)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_out)
        )

        self._reset_parameters()


    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: the sequence to the encoder (required).
            src_mask: the additive mask for the src sequence (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`, `(N, S, E)` if batch_first.
            - src_mask: :math:`(S, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            where S is the source sequence length, T is the target sequence length, N is the
        batch size, E is the feature number
        """

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        memory = memory.mean(dim=1) if self.pool == 'mean' else memory[:, -1, :]
        memory = self.to_latent(memory)
        memory = torch.tanh(self.mlp_head(memory))
        return memory

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        r"""TransformerEncoder is a stack of N encoder layers

        Args:
            encoder_layer: an instance of the TransformerEncoderLayer() class (required).
            num_layers: the number of sub-encoder-layers in the encoder (required).
            norm: the layer normalization component (optional).

        Examples::
            >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
            >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
            >>> src = torch.rand(10, 32, 512)
            >>> out = transformer_encoder(src)
        """
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self,  d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False):
        """
        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
            activation: the activation function of the intermediate layer, can be a string
                        ("relu" or "gelu") or a unary callable. Default: relu
            layer_norm_eps: the eps value in layer normalization components (default=1e-5).
            batch_first: If ``True``, then the input and output tensors are provided
                         as (batch, seq, feature). Default: ``False``.
            norm_first: if ``True``, layer norm is done prior to attention and feedforward
                        operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
        Examples::
            >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
            >>> src = torch.rand(32, 10, 512)
            >>> out = encoder_layer(src)
        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

