from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer


def model_factory(config, data):
    task = config['task']
    if hasattr(data, "all_df_dict"):
        feature_dims = {
            feature_name: df.shape[1]
            for feature_name, df in data.all_df_dict.items()
        }
        feat_dim = sum(feature_dims.values())  # backward compatibility
    else:
        feature_dims = None
        feat_dim = data.feature_df.shape[1]

    # data windowing is used when samples don't have a predefined length or the length is too long
    max_seq_len = config['data_window_len'] if config['data_window_len'] is not None else config['max_seq_len']
    if max_seq_len is None:
        try:
            max_seq_len = data.max_seq_len
        except AttributeError as x:
            print("Data class does not define a maximum sequence length, so it must be defined with the script argument `max_seq_len`")
            raise x

    if (task == "imputation") or (task == "transduction"):
        if config['model'] == 'LINEAR':
            return DummyTSTransformerEncoder(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                                             config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
                                             pos_encoding=config['pos_encoding'], activation=config['activation'],
                                             norm=config['normalization_layer'], freeze=config['freeze'])
        elif config['model'] == 'transformer':
            return TSTransformerEncoder(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                                        config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
                                        pos_encoding=config['pos_encoding'], activation=config['activation'],
                                        norm=config['normalization_layer'], freeze=config['freeze'])

    if (task == "classification") or (task == "regression"):
        num_labels = len(data.class_names) if task == "classification" else data.labels_df.shape[1]  # dimensionality of labels
        if config['model'] == 'LINEAR':
            return DummyTSTransformerEncoderClassiregressor(feat_dim, max_seq_len, config['d_model'],
                                                            config['num_heads'],
                                                            config['num_layers'], config['dim_feedforward'],
                                                            num_classes=num_labels,
                                                            dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                                            activation=config['activation'],
                                                            norm=config['normalization_layer'], freeze=config['freeze'])
        elif config['model'] == 'transformer':
            if feature_dims is not None:
                print("Using multimodal transformer model since data has multiple feature sets: {}".format(feature_dims))
                return MultiModalTSTransformerEncoderClassiregressor(
                    feature_dims=feature_dims,
                    max_len=max_seq_len,
                    d_model=config['d_model'],
                    n_heads=config['num_heads'],
                    num_layers=config['num_layers'],
                    dim_feedforward=config['dim_feedforward'],
                    num_classes=num_labels,
                    dropout=config['dropout'],
                    pos_encoding=config['pos_encoding'],
                    activation=config['activation'],
                    norm=config['normalization_layer'],
                    freeze=config['freeze'],
                    fusion_type=config.get('fusion_type', 'same_time'),
                    use_loss_fusion=config.get('use_loss_fusion', False),
                    lambda_aux=config.get('lambda_aux', 0.2),
                )

            else:
                return TSTransformerEncoderClassiregressor(
                    feat_dim,
                    max_seq_len,
                    config['d_model'],
                    config['num_heads'],
                    config['num_layers'],
                    config['dim_feedforward'],
                    num_classes=num_labels,
                    dropout=config['dropout'],
                    pos_encoding=config['pos_encoding'],
                    activation=config['activation'],
                    norm=config['normalization_layer'],
                    freeze=config['freeze'],
                )
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, is_causal: bool = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output


class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding


        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output
    

class SameTimeCrossModalFusionBlock(nn.Module):
    """
    Same-time cross-modal attention.

    For each time t:
        [eye_t, flow_t] attend to each other.

    Input:
        h_dict[name]: [B, T, d_model]

    Output:
        h_dict[name]: [B, T, d_model]
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(SameTimeCrossModalFusionBlock, self).__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_dict, padding_masks=None):
        feature_names = list(h_dict.keys())

        B, T, D = h_dict[feature_names[0]].shape
        M = len(feature_names)

        # [B, T, M, D]
        tokens = torch.stack([h_dict[name] for name in feature_names], dim=2)

        # [B*T, M, D]
        tokens = tokens.reshape(B * T, M, D)

        fused_tokens, _ = self.attn(
            query=tokens,
            key=tokens,
            value=tokens,
            need_weights=False,
        )

        fused_tokens = self.norm(tokens + self.dropout(fused_tokens))

        # [B, T, M, D]
        fused_tokens = fused_tokens.reshape(B, T, M, D)

        out_dict = {}
        for m, name in enumerate(feature_names):
            out_dict[name] = fused_tokens[:, :, m, :]

        return out_dict
    

class AllTimeCrossModalFusionBlock(nn.Module):
    """
    Cross-modal attention over all modality-time tokens.

    Tokens are:
        eye_1, eye_2, ..., eye_T, flow_1, flow_2, ..., flow_T

    This gives full attention over modality x time.

    Input:
        h_dict[name]: [B, T, d_model]

    Output:
        h_dict[name]: [B, T, d_model]
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(AllTimeCrossModalFusionBlock, self).__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_dict, padding_masks=None):
        feature_names = list(h_dict.keys())

        B, T, D = h_dict[feature_names[0]].shape
        M = len(feature_names)

        # [B, M, T, D]
        tokens = torch.stack([h_dict[name] for name in feature_names], dim=1)

        # [B, M*T, D]
        tokens = tokens.reshape(B, M * T, D)

        key_padding_mask = None

        if padding_masks is not None:
            # padding_masks: [B, T], True = valid
            # Repeat for each modality -> [B, M, T] -> [B, M*T]
            valid_mask = padding_masks.unsqueeze(1).repeat(1, M, 1)
            valid_mask = valid_mask.reshape(B, M * T)

            # MultiheadAttention expects True = ignore
            key_padding_mask = ~valid_mask

        fused_tokens, _ = self.attn(
            query=tokens,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        fused_tokens = self.norm(tokens + self.dropout(fused_tokens))

        # [B, M*T, D] -> [B, M, T, D]
        fused_tokens = fused_tokens.reshape(B, M, T, D)

        out_dict = {}
        for m, name in enumerate(feature_names):
            out_dict[name] = fused_tokens[:, m, :, :]

        return out_dict
    

def masked_mean_pool(h, padding_masks):
    """
    h: [B, T, D]
    padding_masks: [B, T], True = valid

    returns:
        z: [B, D]
    """
    mask = padding_masks.unsqueeze(-1).float()
    return (h * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)


class FeatureFusionBlock(nn.Module):
    """
    Feature-token fusion after temporal Transformer.

    Input:
        h_dict[name]: [B, T, d_model]

    Process:
        1. Mask padding.
        2. Convert each modality from [B, T, d_model] to [B, d_model, T].
        3. Treat each latent feature dimension as one token.
        4. Concatenate modality feature tokens:
              [B, M*d_model, T]
        5. Project each feature trajectory T -> d_model.
        6. Transformer over feature tokens.
        7. Pool over feature tokens.

    Output:
        z_feature: [B, d_model]
    """

    def __init__(
        self,
        max_len,
        d_model,
        n_heads,
        num_modalities,
        dim_feedforward,
        dropout=0.1,
        activation='gelu',
    ):
        super(FeatureFusionBlock, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.num_modalities = num_modalities

        # Each latent feature token is represented by its temporal trajectory.
        # Project trajectory length T -> d_model.
        self.time_to_token = nn.Linear(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

        self.feature_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )

        # Optional learnable token embeddings for modality-feature identity
        self.token_embed = nn.Parameter(
            torch.empty(num_modalities * d_model, d_model)
        )
        nn.init.uniform_(self.token_embed, -0.02, 0.02)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_dict, padding_masks):
        """
        Args:
            h_dict:
                dict of [B, T, d_model]

            padding_masks:
                [B, T], True = valid, False = padding

        Returns:
            z_feature:
                [B, d_model]
        """

        feature_names = list(h_dict.keys())

        feature_tokens = []

        for name in feature_names:
            h = h_dict[name]  # [B, T, d_model]

            # Zero-out padded time steps before feature-token construction
            h = h * padding_masks.unsqueeze(-1)

            # [B, T, d_model] -> [B, d_model, T]
            h_feat = h.permute(0, 2, 1)

            feature_tokens.append(h_feat)

        # [B, M*d_model, T]
        feature_tokens = torch.cat(feature_tokens, dim=1)

        # If actual T is shorter than max_len, pad feature trajectory to max_len
        B, num_tokens, T = feature_tokens.shape

        if T < self.max_len:
            pad = torch.zeros(
                B,
                num_tokens,
                self.max_len - T,
                device=feature_tokens.device,
                dtype=feature_tokens.dtype,
            )
            feature_tokens = torch.cat([feature_tokens, pad], dim=-1)

        elif T > self.max_len:
            feature_tokens = feature_tokens[:, :, :self.max_len]

        # [B, M*d_model, T] -> [B, M*d_model, d_model]
        tokens = self.time_to_token(feature_tokens)

        # Add token identity embedding
        tokens = tokens + self.token_embed.unsqueeze(0)

        tokens = self.norm(tokens)
        tokens = self.dropout(tokens)

        # Feature-token attention
        tokens = self.feature_transformer(tokens)  # [B, M*d_model, d_model]

        # Pool over feature tokens
        z_feature = tokens.mean(dim=1)  # [B, d_model]

        return z_feature


class MultiModalTSTransformerEncoderClassiregressor(nn.Module):
    """
    Multimodal Transformer classifier/regressor with selectable fusion.

    fusion_type:
        "same_time":
            temporal encoder per modality
            cross-modal attention at same time step
            concat sequence
            flatten classifier

        "late":
            temporal encoder per modality
            masked mean pooling per modality
            concat pooled vectors
            classifier

        "all_time":
            temporal encoder per modality
            attention over all modality-time tokens
            concat sequence
            flatten classifier

        "gated":
            the first modality keeps the path of a single-modality "same_time" model
            the other modalities only add a correction: same-time fusion -> classifier,
            scaled by a learned gate that starts at 0
            logits = logits_primary + gate * logits_correction
    """

    def __init__(
        self,
        feature_dims,
        max_len,
        d_model,
        n_heads,
        num_layers,
        dim_feedforward,
        num_classes,
        dropout=0.1,
        pos_encoding='fixed',
        activation='gelu',
        norm='BatchNorm',
        freeze=False,
        fusion_type='same_time',
        use_loss_fusion=False,
        lambda_aux=0.2,
    ):
        super(MultiModalTSTransformerEncoderClassiregressor, self).__init__()

        if fusion_type not in ['same_time', 'late', 'all_time', 'feature', 'both', 'gated']:
            raise ValueError(
                f"Unknown fusion_type={fusion_type}. "
                "Choose from ['same_time', 'late', 'all_time', 'feature', 'both', 'gated']."
            )

        # Modules kept frozen and in eval mode, see freeze_features()
        self.frozen_modules = []

        self.feature_dims = feature_dims
        self.feature_names = list(feature_dims.keys())

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_classes = num_classes
        self.num_modalities = len(self.feature_names)
        self.fusion_type = fusion_type
        if use_loss_fusion == "True":
            self.use_loss_fusion = True
        else:
            self.use_loss_fusion = False
        self.lambda_aux = lambda_aux

        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)

        # One input projection per modality
        self.project_inp = nn.ModuleDict({
            feature_name: nn.Linear(feature_dim, d_model)
            for feature_name, feature_dim in feature_dims.items()
        })

        # One positional encoder per modality
        self.pos_enc = nn.ModuleDict({
            feature_name: get_pos_encoder(pos_encoding)(
                d_model,
                dropout=dropout * (1.0 - freeze),
                max_len=max_len,
            )
            for feature_name in self.feature_names
        })

        # One temporal transformer encoder per modality
        self.temporal_encoders = nn.ModuleDict()

        for feature_name in self.feature_names:
            if norm == 'LayerNorm':
                encoder_layer = TransformerEncoderLayer(
                    d_model,
                    n_heads,
                    dim_feedforward,
                    dropout * (1.0 - freeze),
                    activation=activation,
                )
            else:
                encoder_layer = TransformerBatchNormEncoderLayer(
                    d_model,
                    n_heads,
                    dim_feedforward,
                    dropout * (1.0 - freeze),
                    activation=activation,
                )

            self.temporal_encoders[feature_name] = nn.TransformerEncoder(
                encoder_layer,
                num_layers,
            )

        # Fusion block
        if self.fusion_type == 'same_time':
            self.fusion_block = SameTimeCrossModalFusionBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )

            self.feature_fusion_block = None
            output_dim = self.num_modalities * d_model * max_len

        elif self.fusion_type == 'all_time':
            self.fusion_block = AllTimeCrossModalFusionBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )

            self.feature_fusion_block = None
            output_dim = self.num_modalities * d_model * max_len

        elif self.fusion_type == 'late':
            self.fusion_block = None
            self.feature_fusion_block = None

            output_dim = self.num_modalities * d_model

        elif self.fusion_type == 'feature':
            self.fusion_block = None

            self.feature_fusion_block = FeatureFusionBlock(
                max_len=max_len,
                d_model=d_model,
                n_heads=n_heads,
                num_modalities=self.num_modalities,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )

            output_dim = d_model

        elif self.fusion_type == 'both':
            # Time path: same-time cross-modal fusion
            self.fusion_block = SameTimeCrossModalFusionBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )

            # Feature path: latent-feature fusion
            self.feature_fusion_block = FeatureFusionBlock(
                max_len=max_len,
                d_model=d_model,
                n_heads=n_heads,
                num_modalities=self.num_modalities,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )

            # same-time path output: [B, T, M*d_model] -> flatten
            # feature path output:   [B, d_model]
            output_dim = self.num_modalities * d_model * max_len + d_model

        elif self.fusion_type == 'gated':
            # Primary path: same layers and names as a single-modality "same_time" model,
            # so its fusion block and classifier can be loaded from such a model
            self.primary_feature = self.feature_names[0]

            self.primary_fusion_block = SameTimeCrossModalFusionBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )

            self.primary_output_layer = nn.Sequential(
                nn.Linear(d_model * max_len, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )

            # Correction path: same-time cross-modal fusion over all modalities
            self.fusion_block = SameTimeCrossModalFusionBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )

            self.feature_fusion_block = None
            self.gate = nn.Parameter(torch.zeros(1))
            output_dim = self.num_modalities * d_model * max_len

        self.output_layer = nn.Sequential(
            nn.Linear(output_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self.aux_output_layers = nn.ModuleDict({
            feature_name: nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )
            for feature_name in self.feature_names
        })

    
    def branch_modules(self, feature_name):
        """
        Modules that belong to one modality only.
        """
        modules = [
            self.project_inp[feature_name],
            self.pos_enc[feature_name],
            self.temporal_encoders[feature_name],
            self.aux_output_layers[feature_name],
        ]

        if self.fusion_type == 'gated' and feature_name == self.primary_feature:
            modules += [self.primary_fusion_block, self.primary_output_layer]

        return modules

    def freeze_features(self, feature_names):
        """
        Stop training the branch of each given modality and keep it in eval mode,
        so its weights, batch-norm statistics and (absent) dropout stay as loaded.
        """
        for feature_name in feature_names:
            if feature_name not in self.feature_names:
                raise ValueError(
                    f"Cannot freeze '{feature_name}': not one of the model's features {self.feature_names}."
                )

            for module in self.branch_modules(feature_name):
                for param in module.parameters():
                    param.requires_grad = False
                self.frozen_modules.append(module)

        self.train(self.training)

    def train(self, mode=True):
        super(MultiModalTSTransformerEncoderClassiregressor, self).train(mode)

        for module in self.frozen_modules:
            module.eval()

        return self

    def load_pretrained_branches(self, state_dict):
        """
        Copy the per-modality weights of a previously trained model into this one.
        Branches are matched by feature name, everything else is left untouched.

        With "gated" fusion, the fusion block and classifier of a single-modality
        model of the primary feature are loaded into the primary path.

        Returns:
            loaded features, and whether the primary classifier was loaded
        """
        groups = ['project_inp', 'pos_enc', 'temporal_encoders', 'aux_output_layers']
        source_features = sorted({key.split('.')[1] for key in state_dict if key.startswith('project_inp.')})

        load_primary_head = (
            self.fusion_type == 'gated'
            and source_features == [self.primary_feature]
        )

        own_state = self.state_dict()
        new_state = {}
        loaded_features = set()

        for key, value in state_dict.items():
            target = None

            for feature_name in self.feature_names:
                if any(key.startswith(f"{group}.{feature_name}.") for group in groups):
                    target = key
                    loaded_features.add(feature_name)

            if load_primary_head and (key.startswith('fusion_block.') or key.startswith('output_layer.')):
                target = 'primary_' + key

            if target is None:
                continue

            if target not in own_state or own_state[target].shape != value.shape:
                raise ValueError(
                    f"Pre-trained weight '{key}' does not fit this model (d_model, layers or sequence length differ)."
                )

            new_state[target] = value

        self.load_state_dict(new_state, strict=False)

        return sorted(loaded_features), load_primary_head

    def compute_aux_logits(self, h_dict, padding_masks):
        """
        Compute one auxiliary prediction per modality.

        Args:
            h_dict[name]: [B, T, d_model]
            padding_masks: [B, T], True = valid

        Returns:
            aux_logits:
                {
                    "eyeFeat": logits_eye,
                    "opticalFlow": logits_flow,
                    ...
                }
        """

        aux_logits = {}

        for feature_name in self.feature_names:
            z_i = masked_mean_pool(
                h_dict[feature_name],
                padding_masks,
            )  # [B, d_model]

            aux_logits[feature_name] = self.aux_output_layers[feature_name](z_i)

        return aux_logits


    def encode_modalities(self, X, padding_masks):
        """
        Encode each modality separately with its own temporal Transformer.

        Args:
            X[name]: [B, T, D_i]
            padding_masks: [B, T]

        Returns:
            h_dict[name]: [B, T, d_model]
        """

        h_dict = {}

        for feature_name in self.feature_names:
            x_i = X[feature_name]  # [B, T, D_i]

            # Transformer expects [T, B, D]
            inp = x_i.permute(1, 0, 2)

            inp = self.project_inp[feature_name](inp) * math.sqrt(self.d_model)
            inp = self.pos_enc[feature_name](inp)

            out = self.temporal_encoders[feature_name](
                inp,
                src_key_padding_mask=~padding_masks,
            )

            out = self.act(out)
            out = out.permute(1, 0, 2)  # [B, T, d_model]
            out = self.dropout1(out)

            h_dict[feature_name] = out

        return h_dict

    def forward(self, X, padding_masks):
        """
        X:
            dict[str, Tensor[B, T, D_i]]

        padding_masks:
            Tensor[B, T], True = valid, False = padding

        returns:
            Tensor[B, num_classes]
        """

        h_dict = self.encode_modalities(X, padding_masks)

        # --------------------------------------------------
        # Option 1: same-time fusion
        # eye[t] attends to flow[t]
        # --------------------------------------------------
        if self.fusion_type == 'same_time':
            h_dict = self.fusion_block(h_dict, padding_masks)

            for feature_name in self.feature_names:
                h_dict[feature_name] = h_dict[feature_name] * padding_masks.unsqueeze(-1)

            fused = torch.cat(
                [h_dict[name] for name in self.feature_names],
                dim=-1,
            )  # [B, T, M*d_model]

            fused = fused.reshape(fused.shape[0], -1)

        # --------------------------------------------------
        # Option 2: all-time fusion
        # eye[t] can attend to flow[t']
        # --------------------------------------------------
        elif self.fusion_type == 'all_time':
            h_dict = self.fusion_block(h_dict, padding_masks)

            for feature_name in self.feature_names:
                h_dict[feature_name] = h_dict[feature_name] * padding_masks.unsqueeze(-1)

            fused = torch.cat(
                [h_dict[name] for name in self.feature_names],
                dim=-1,
            )  # [B, T, M*d_model]

            fused = fused.reshape(fused.shape[0], -1)

        # --------------------------------------------------
        # Option 3: late fusion
        # pool each modality, then concatenate
        # --------------------------------------------------
        elif self.fusion_type == 'late':
            pooled = []

            for feature_name in self.feature_names:
                z_i = masked_mean_pool(
                    h_dict[feature_name],
                    padding_masks,
                )  # [B, d_model]

                pooled.append(z_i)

            fused = torch.cat(pooled, dim=-1)  # [B, M*d_model]

        # --------------------------------------------------
        # Option 4: feature fusion
        # attend over latent feature tokens after temporal Transformer
        # --------------------------------------------------
        elif self.fusion_type == 'feature':
            fused = self.feature_fusion_block(
                h_dict,
                padding_masks,
            )  # [B, d_model]

        # --------------------------------------------------
        # Option 5: both fusion
        # temporal pooled fusion + feature-token fusion
        # --------------------------------------------------
        elif self.fusion_type == 'both':
            # --------------------------------------------------
            # Path 1: same-time cross-modal temporal fusion
            # --------------------------------------------------
            h_same_time = self.fusion_block(h_dict, padding_masks)

            for feature_name in self.feature_names:
                h_same_time[feature_name] = (
                    h_same_time[feature_name] * padding_masks.unsqueeze(-1)
                )

            z_same_time = torch.cat(
                [h_same_time[name] for name in self.feature_names],
                dim=-1,
            )  # [B, T, M*d_model]

            z_same_time = z_same_time.reshape(z_same_time.shape[0], -1)
            # [B, T*M*d_model]

            # --------------------------------------------------
            # Path 2: feature-token fusion
            # Use original h_dict, not h_same_time, to keep a clean feature path.
            # --------------------------------------------------
            z_feature = self.feature_fusion_block(
                h_dict,
                padding_masks,
            )  # [B, d_model]

            # --------------------------------------------------
            # Combine both paths
            # --------------------------------------------------
            fused = torch.cat(
                [z_same_time, z_feature],
                dim=-1,
            )

        # --------------------------------------------------
        # Option 6: gated fusion
        # primary modality alone + gate * cross-modal correction
        # --------------------------------------------------
        elif self.fusion_type == 'gated':
            primary = self.primary_feature

            h_primary = self.primary_fusion_block(
                {primary: h_dict[primary]},
                padding_masks,
            )[primary]
            h_primary = h_primary * padding_masks.unsqueeze(-1)
            logits_primary = self.primary_output_layer(
                h_primary.reshape(h_primary.shape[0], -1)
            )

            h_dict = self.fusion_block(h_dict, padding_masks)

            for feature_name in self.feature_names:
                h_dict[feature_name] = h_dict[feature_name] * padding_masks.unsqueeze(-1)

            fused = torch.cat(
                [h_dict[name] for name in self.feature_names],
                dim=-1,
            )  # [B, T, M*d_model]

            fused = fused.reshape(fused.shape[0], -1)

        else:
            raise ValueError(f"Unknown fusion_type={self.fusion_type}")

        logits_fused = self.output_layer(fused)

        if self.fusion_type == 'gated':
            logits_fused = logits_primary + self.gate * logits_fused

        if self.use_loss_fusion and self.training:
            aux_logits = self.compute_aux_logits(
                h_dict=h_dict,
                padding_masks=padding_masks,
            )

            output = {"fused": logits_fused}
            output.update(aux_logits)

            return output

        return logits_fused